import os
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from typing import List, Dict, Tuple, Any

import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


class DeepSeekVL2Evaluator:
    """
    Evaluator adapting the official DeepSeek-VL2 inference pipeline.
    Key differences from VL1:
      - Package name: deepseek_vl2
      - Image placeholder: <image>  (VL1 used <image_placeholder>)
      - Role names: <|User|> / <|Assistant|>  (VL1 used User / Assistant)
      - processor() call requires a system_prompt argument
      - Each image in multi-image input needs its own <image> token
    """

    def __init__(
        self,
        model_path: str,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    ):
        self.processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(
            model_path
        )
        self.tokenizer = self.processor.tokenizer

        self.model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
        )
        self.model = self.model.to(device_map).eval()
        self.device = next(self.model.parameters()).device

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_inputs(
        self,
        text: str,
        img_paths: List[str],
        system_prompt: str = "",
    ) -> Dict[str, Any]:
        conversation = [
            {
                "role": "<|User|>",
                "content": text,
                "images": img_paths,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        pil_images = load_pil_images(conversation)
        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=system_prompt,
        ).to(self.device)

        return prepare_inputs

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def choice_single_img_promote(
        self,
        promote: str,
        question: str,
        options: List[str],
        img_path: str,
        language_type: str = "cn",
    ) -> Dict[str, Any]:
        n = len(options)
        assert n > 0, "Need at least one option."
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n]
        options_text = "\n".join(f"{letters[i]}. {options[i]}" for i in range(n))

        if language_type == "cn":
            text_prompt = (
                f"这里有 {n} 个候选项, 每一个是一段文字\n"
                f"问题: {question}\n\n"
                f"选项:\n{options_text}\n\n"
                f"请直接输出正确选项的字母（如A、B、C），不要解释。\n"
            )
        else:
            text_prompt = (
                f"There are {n} candidate options, each one is a text.\n"
                f"Question: {question}\n\n"
                f"Options:\n{options_text}\n\n"
                f"Output only the letter of the correct option (e.g. A). Do not explain.\n"
            )

        # Single image: one <image> tag before the prompt text.
        content = f"<image>\n{promote}{text_prompt}"
        return self._create_inputs(content, [img_path])

    def choice_multi_img_promote(
        self,
        promote: str,
        question: str,
        img_paths: List[str],
        language_type: str = "cn",
    ) -> Tuple[Dict[str, Any], List[str]]:
        n = len(img_paths)
        assert n > 0, "Need at least one choice image."
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n])
        mapping_str = ", ".join(f"image {i+1} = option {letters[i]}" for i in range(n))

        # Each image needs its own <image> placeholder in the content string.
        img_tags = "".join(
            f"{letters[i]}: <image>\n" for i in range(n)
        )

        if language_type == "cn":
            text_prompt = (
                f"这里有 {n} 个候选项, 每一个是一张图片\n"
                f"映射关系为: {mapping_str}.\n\n"
                f"问题: {question}\n\n"
                f"请直接输出正确选项的字母（如A、B、C），不要解释。\n"
            )
        else:
            text_prompt = (
                f"There are {n} candidate options, each one is an image.\n"
                f"The mapping is: {mapping_str}.\n\n"
                f"Question: {question}\n\n"
                f"Output only the letter of the correct option (e.g. A). Do not explain.\n"
            )

        content = f"{promote}{img_tags}{text_prompt}"
        inputs = self._create_inputs(content, img_paths)
        return inputs, letters

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_text(
        self, prepare_inputs: Dict[str, Any], max_new_tokens: int = 64
    ) -> str:
        with torch.inference_mode():
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
            outputs = self.model.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        answer = self.tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=True
        )
        # BPE byte-to-unicode artefacts (same as VL1).
        answer = answer.replace("Ġ", " ").replace("Ċ", "\n")
        # Strip hallucinated multi-turn continuation.
        for stop_str in ("\n<|User|>", "\n<|Assistant|>", "\nUser:", "\nuser:"):
            idx = answer.find(stop_str)
            if idx != -1:
                answer = answer[:idx]

        del outputs, inputs_embeds
        torch.cuda.empty_cache()

        return answer.strip()

    # ------------------------------------------------------------------
    # Public eval helpers
    # ------------------------------------------------------------------

    def eval_choice_single_img(
        self,
        promote: str,
        img_path: str,
        question: str,
        options: List[str],
        max_new_tokens: int = 8,
        language_type: str = "cn",
    ) -> Tuple[str, str]:
        inputs = self.choice_single_img_promote(
            promote, question, options, img_path, language_type
        )
        out = self.generate_text(inputs, max_new_tokens=max_new_tokens)
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(options)]
        m = re.search(r"(?<![A-Za-z])([" + letters + r"])(?=[^A-Za-z]|$)", out)
        return out, m.group(1) if m else None

    def eval_choice_multi_img(
        self,
        promote: str,
        img_paths: List[str],
        question: str,
        max_new_tokens: int = 8,
        language_type: str = "cn",
    ) -> Tuple[str, str]:
        inputs, letters = self.choice_multi_img_promote(
            promote, question, img_paths, language_type
        )
        out = self.generate_text(inputs, max_new_tokens=max_new_tokens)
        pattern = r"(?<![A-Za-z])([" + "".join(letters) + r"])(?=[^A-Za-z]|$)"
        m = re.search(pattern, out)
        return out, m.group(1) if m else None


if __name__ == "__main__":
    model_path = str(config.MODEL_ROOT / "deepseek-vl2-small")
    img_data_root = str(config.IMAGE_DATA_DIR)

    evaluator = DeepSeekVL2Evaluator(model_path)
    promote = ""

    def _first_img(site: str) -> str:
        d = os.path.join(img_data_root, site)
        imgs = sorted(
            f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        return os.path.join(d, imgs[0])

    img_single = _first_img("Ancient_Building_Complex_in_the_Wudang_Mountains")

    raw1, choice1 = evaluator.eval_choice_single_img(
        promote=promote,
        img_path=img_single,
        question="What type of cultural heritage site is shown in the image?",
        options=["Royal garden", "Taoist holy site", "Buddhist temple", "Confucian academy"],
        max_new_tokens=16,
        language_type="cn",
    )
    print("=== Case 1: Single image / CN ===")
    print("raw   :", raw1)
    print("choice:", choice1, " (expected: B)\n")

    raw1e, choice1e = evaluator.eval_choice_single_img(
        promote=promote,
        img_path=img_single,
        question="What type of cultural heritage site is shown in the image?",
        options=["Royal garden", "Taoist holy site", "Buddhist temple", "Confucian academy"],
        max_new_tokens=8,
        language_type="en",
    )
    print("=== Case 1: Single image / EN ===")
    print("raw   :", raw1e)
    print("choice:", choice1e, " (expected: B)\n")

    multi_imgs = [
        _first_img("Ancient_Building_Complex_in_the_Wudang_Mountains"),
        _first_img("Classical_Gardens_of_Suzhou"),
        _first_img("Ancient_City_of_Ping_Yao"),
        _first_img("Cultural_Landscape_of_Honghe_Hani_Rice_Terraces"),
    ]

    raw2, choice2 = evaluator.eval_choice_multi_img(
        promote=promote,
        img_paths=multi_imgs,
        question="Which image shows a rice terrace landscape?",
        max_new_tokens=16,
        language_type="cn",
    )
    print("=== Case 2: Multi image / CN ===")
    print("raw   :", raw2)
    print("choice:", choice2, " (expected: D)\n")

    raw2e, choice2e = evaluator.eval_choice_multi_img(
        promote=promote,
        img_paths=multi_imgs,
        question="Which image shows a rice terrace landscape?",
        max_new_tokens=8,
        language_type="en",
    )
    print("=== Case 2: Multi image / EN ===")
    print("raw   :", raw2e)
    print("choice:", choice2e, " (expected: D)\n")
