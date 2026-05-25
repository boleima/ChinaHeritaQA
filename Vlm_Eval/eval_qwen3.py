import os
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from typing import List, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor

# Qwen3-VL: prefer Qwen3VLForConditionalGeneration (transformers >= 4.52.0);
# fall back to Qwen2_5_VLForConditionalGeneration for older environments.
try:
    from transformers import Qwen3VLForConditionalGeneration as _ModelCls
except ImportError:
    from transformers import Qwen2_5_VLForConditionalGeneration as _ModelCls

try:
    from qwen_vl_utils import process_vision_info as _process_vision_info
    _HAS_QWEN_VL_UTILS = True
except ImportError:
    _HAS_QWEN_VL_UTILS = False

# Strips <think>...</think> blocks that Qwen3-VL emits when thinking is enabled.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _get_vision_inputs(messages):
    if _HAS_QWEN_VL_UTILS:
        return _process_vision_info(messages)
    images = []
    for msg in messages:
        for item in msg.get("content", []):
            if item.get("type") != "image":
                continue
            img = item.get("image")
            if img is None:
                continue
            if isinstance(img, str):
                images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                images.append(img)
    return (images if images else None), None


class Qwen3VLEvaluator:
    def __init__(
        self,
        model_path: str,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        enable_thinking: bool = False,
    ):
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self.processor.image_processor.min_pixels = min_pixels
        self.processor.image_processor.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        # When False, thinking tokens are suppressed via chat-template flag and
        # any residual <think> blocks are stripped from decoded output.
        self.enable_thinking = enable_thinking

        try:
            self.model = _ModelCls.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                attn_implementation="flash_attention_2",
            )
        except (ImportError, ValueError):
            self.model = _ModelCls.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
            )
        self.device = next(self.model.parameters()).device

    def _apply_template(self, messages) -> str:
        # Qwen3-VL's chat template accepts enable_thinking to control CoT output.
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # Older processor versions may not support enable_thinking.
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def choice_single_img_promote(
        self,
        promote: str,
        question: str,
        options: List[str],
        img_path: str,
        language_type: str = "cn",
    ):
        n = len(options)
        assert n > 0, "Need at least one option."
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n]
        options_text = "\n".join(f"{letters[i]}. {options[i]}" for i in range(n))

        if language_type == "cn":
            text_prompt = (
                f"这里有 {n} 个候选项, 每一个是一段文字\n"
                f"问题: {question}\n\n选项:\n{options_text}\n\n"
            )
        else:
            text_prompt = (
                f"There are {n} candidate options, each one is a text.\n"
                f"Question: {question}\n\nOptions:\n{options_text}\n\n"
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": f"{promote}{text_prompt}"},
                ],
            }
        ]

        text = self._apply_template(messages)
        image_inputs, video_inputs = _get_vision_inputs(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        return inputs

    def choice_multi_img_promote(
        self,
        promote: str,
        question: str,
        img_paths: List[str],
        language_type: str = "cn",
    ):
        n = len(img_paths)
        assert n > 0, "Need at least one choice image."
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n])
        mapping_str = ", ".join(f"image {i+1} = option {letters[i]}" for i in range(n))

        if language_type == "cn":
            text_prompt = (
                f"这里有 {n} 个候选项, 每一个是一张图片\n"
                f"映射关系为: {mapping_str}.\n\n问题: {question}\n\n"
            )
        else:
            text_prompt = (
                f"There are {n} candidate options, each one is an image.\n"
                f"The mapping is: {mapping_str}.\n\nQuestion: {question}\n\n"
            )

        content_list = [
            {
                "type": "image",
                "image": p,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            }
            for p in img_paths
        ]
        content_list.append({"type": "text", "text": f"{promote}{text_prompt}"})

        messages = [{"role": "user", "content": content_list}]

        text = self._apply_template(messages)
        image_inputs, video_inputs = _get_vision_inputs(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        return inputs, letters

    def generate_text(self, inputs, max_new_tokens: int = 64) -> str:
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        result = output_text[0].strip()

        # Remove thinking blocks regardless of enable_thinking, so callers always
        # receive clean text.
        result = _THINK_RE.sub("", result).strip()

        del generated_ids, generated_ids_trimmed, inputs
        torch.cuda.empty_cache()

        return result

    @staticmethod
    def _extract_letter(text: str) -> str:
        m = re.search(r'\b([A-Z])\b', text)
        if m:
            return m.group(1)
        m = re.search(r'[A-Z]', text)
        return m.group(0) if m else ""

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
        return out, self._extract_letter(out)

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
        return out, self._extract_letter(out)


if __name__ == "__main__":
    model_path = str(config.MODEL_ROOT / "Qwen3-VL-8B-Instruct")
    img_root   = str(config.IMAGE_DATA_DIR / "Ancient_Building_Complex_in_the_Wudang_Mountains")
    files = os.listdir(img_root)
    choice_paths = [os.path.join(img_root, f) for f in files[:4]]

    # enable_thinking=False gives direct letter outputs, which is suitable for MCQ.
    evaluator = Qwen3VLEvaluator(model_path, enable_thinking=False)
    promote = (
        "You are a professional cultural and natural heritage recognition system. "
        "Please carefully examine the image and read through options A, B, C, D, and E, "
        "then identify the single correct answer based on the visual content.\n"
        "Requirements:\n- Output only the letter of the correct option (A / B / C / D / E)\n"
        "- Do not include any explanation, punctuation, or additional content\nExample answer: A\n"
    )

    question1 = "What is in this image?"
    options1 = ["a modern city", "an ancient temple", "a beach", "a forest"]

    raw1, choice1 = evaluator.eval_choice_single_img(
        promote, choice_paths[0], "What is contained in this image?", options1, language_type="cn"
    )
    print("Single-image MC raw:", raw1)
    print("Single-image MC choice:", choice1)

    raw1, choice1 = evaluator.eval_choice_single_img(
        promote, choice_paths[0], question1, options1, language_type="en"
    )
    print("Single-image MC raw:", raw1)
    print("Single-image MC choice:", choice1)

    question2 = "Which image contains red color?"
    raw2, choice2 = evaluator.eval_choice_multi_img(
        promote, choice_paths, question2, max_new_tokens=1024, language_type="cn"
    )
    print("Multi-image MC raw:", raw2)
    print("Multi-image MC choice:", choice2)
