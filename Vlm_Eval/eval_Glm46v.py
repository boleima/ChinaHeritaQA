import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# Set as early as possible, before importing torch.
# Helps reduce CUDA allocator fragmentation for multi-image, variable-size, and long-context workloads.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import gc
import re
import warnings
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Glm46VForConditionalGeneration

try:
    from transformers import LogitsProcessor, LogitsProcessorList
except ImportError:
    from transformers.generation import LogitsProcessor, LogitsProcessorList


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class ForceChoiceTokens(LogitsProcessor):
    """
    Restrict next-token generation to only the specified token IDs.
    Used for MCQ evaluation so the model can only output A/B/C/D/...
    and cannot emit <think> tokens.
    """

    def __init__(self, allowed_token_ids: List[int]):
        if not allowed_token_ids:
            raise ValueError("allowed_token_ids cannot be empty.")
        self.allowed_token_ids = sorted(set(int(x) for x in allowed_token_ids))

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed_token_ids] = scores[:, self.allowed_token_ids]
        return mask


class GLM46VFlashEvaluator:
    def __init__(
        self,
        model_path: str,
        dtype=torch.bfloat16,
        trust_remote_code: bool = True,
        device: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        max_image_side: Optional[int] = 1024,
        allow_eager_fallback: bool = True,
    ):
        """
        GLM-4.6V-Flash multiple-choice evaluator.

        Key design decisions:
        1. Uses SDPA by default instead of FlashAttention or eager.
        2. Does not use chat_template_kwargs / enable_thinking because the processor ignores it.
        3. MCQ uses constrained decoding to generate a single answer token.
        4. use_cache=False by default to reduce VRAM usage.
        5. Optional image resize to lower OOM risk with multiple images.
        """

        self.max_image_side = max_image_side

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if attn_implementation is None:
            # When FlashAttention is unavailable, prefer PyTorch SDPA.
            # Do not default to eager.
            attn_implementation = "sdpa" if torch.cuda.is_available() else "eager"

        self.attn_implementation = attn_implementation

        self.model = self._load_model(
            model_path=model_path,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            allow_eager_fallback=allow_eager_fallback,
        )

        self.model = self.model.to(device).eval()
        self.device = next(self.model.parameters()).device

        print(f"[INFO] device = {self.device}")
        print(f"[INFO] dtype = {dtype}")
        print(f"[INFO] attn_implementation = {self.attn_implementation}")
        print(f"[INFO] max_image_side = {self.max_image_side}")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _from_pretrained_with_dtype_fallback(
        self,
        model_path: str,
        dtype,
        kwargs: Dict,
    ):
        """
        Newer transformers versions use dtype=..., while some older
        versions still use torch_dtype=... This helper handles both.
        """

        try:
            return Glm46VForConditionalGeneration.from_pretrained(
                model_path,
                dtype=dtype,
                **kwargs,
            )
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise

            return Glm46VForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                **kwargs,
            )

    def _load_model(
        self,
        model_path: str,
        dtype,
        trust_remote_code: bool,
        attn_implementation: str,
        allow_eager_fallback: bool,
    ):
        load_kwargs = dict(
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )

        try:
            return self._from_pretrained_with_dtype_fallback(
                model_path=model_path,
                dtype=dtype,
                kwargs=load_kwargs,
            )

        except Exception as exc:
            if attn_implementation != "eager" and allow_eager_fallback:
                warnings.warn(
                    f"Loading with attn_implementation={attn_implementation} failed. "
                    f"Retrying with eager attention. "
                    f"Original error: {repr(exc)}"
                )

                self.attn_implementation = "eager"
                load_kwargs["attn_implementation"] = "eager"

                return self._from_pretrained_with_dtype_fallback(
                    model_path=model_path,
                    dtype=dtype,
                    kwargs=load_kwargs,
                )

            raise

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _load_image(self, img_path: str) -> Image.Image:
        img = Image.open(img_path).convert("RGB")

        if self.max_image_side is not None:
            w, h = img.size
            max_side = max(w, h)

            if max_side > self.max_image_side:
                scale = self.max_image_side / max_side
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                img = img.resize((new_w, new_h), Image.BICUBIC)

        return img

    # ------------------------------------------------------------------
    # Processor / template helpers
    # ------------------------------------------------------------------

    def _move_inputs_to_device(self, inputs):
        return inputs.to(self.device)

    def _apply_template(self, messages: list):
        """
        Does not pass chat_template_kwargs.
        The GLM-4.6V-Flash processor ignores chat_template_kwargs,
        so enable_thinking=False cannot be relied upon here.
        """

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

        return self._move_inputs_to_device(inputs)

    def _create_inputs_single(
        self,
        text: str,
        img: Image.Image,
    ):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": text},
                ],
            }
        ]

        return self._apply_template(messages)

    def _create_inputs_multi(
        self,
        text: str,
        imgs: List[Image.Image],
    ):
        content = [{"type": "image", "image": img} for img in imgs]
        content.append({"type": "text", "text": text})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        return self._apply_template(messages)

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def _get_tokenizer(self):
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("processor has no tokenizer.")
        return tokenizer

    def _decode(self, token_ids: List[int]) -> str:
        if hasattr(self.processor, "decode"):
            return self.processor.decode(
                token_ids,
                skip_special_tokens=True,
            )

        tokenizer = self._get_tokenizer()
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )

    def _choice_token_map(self, letters: List[str]) -> Dict[int, str]:
        """
        Build a token_id → option letter mapping.
        All returned values are normalized to a single uppercase letter.
        """

        tokenizer = self._get_tokenizer()
        token_to_letter: Dict[int, str] = {}

        for letter in letters:
            candidates = [
                letter,
                " " + letter,
                "\n" + letter,
            ]

            for candidate in candidates:
                ids = tokenizer.encode(
                    candidate,
                    add_special_tokens=False,
                )

                if len(ids) == 1:
                    token_to_letter[int(ids[0])] = letter

        if not token_to_letter:
            raise RuntimeError(
                f"Could not build one-token choice map for letters: {letters}"
            )

        return token_to_letter

    @staticmethod
    def _extract_choice_free_text(
        output: str,
        letters: List[str],
    ) -> Optional[str]:
        """
        Fallback for free-generation debug only.
        Formal MCQ evaluation uses constrained decoding, not this method.
        """

        box = re.search(
            r"<\|begin_of_box\|>\s*([A-Z])\s*<\|end_of_box\|>",
            output,
        )
        if box and box.group(1) in letters:
            return box.group(1)

        pattern = r"(?<![A-Za-z])([" + "".join(letters) + r"])(?=[^A-Za-z]|$)"
        match = re.search(pattern, output)

        return match.group(1) if match else None

    # ------------------------------------------------------------------
    # Prompt builders (Updated with Fake CoT)
    # ------------------------------------------------------------------

    def choice_single_img_promote(
        self,
        promote: str,
        question: str,
        options: List[str],
        img: Image.Image,
        language_type: str = "cn",
    ):
        n = len(options)
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n]

        options_text = "\n".join(
            f"{letters[i]}. {options[i]}" for i in range(n)
        )

        # Inject a fake CoT block and forced output suffix to prevent distribution collapse under constrained decoding
        if language_type == "cn":
            text_prompt = (
                f"这里有 {n} 个候选项，每一个是一段文字。\n"
                f"问题：{question}\n\n"
                f"选项：\n{options_text}\n\n"
                f"<think>\n我已经仔细比对了图片内容与各个选项，思考完毕。\n</think>\n"
                f"根据以上分析，正确选项只有一个大写字母。正确选项是："
            )
        else:
            text_prompt = (
                f"There are {n} candidate options, each one is a text.\n"
                f"Question: {question}\n\n"
                f"Options:\n{options_text}\n\n"
                f"<think>\nI have carefully analyzed the image and options. Thinking process complete.\n</think>\n"
                f"Based on the analysis, the single correct uppercase option letter is: "
            )

        return self._create_inputs_single(
            text=promote + text_prompt,
            img=img,
        )

    def choice_multi_img_promote(
        self,
        promote: str,
        question: str,
        choice_imgs: List[Image.Image],
        language_type: str = "cn",
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        n = len(choice_imgs)
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n])

        mapping_str = ", ".join(
            f"image {i + 1} = option {letters[i]}" for i in range(n)
        )

        # Inject fake CoT block and forced output suffix
        if language_type == "cn":
            text_prompt = (
                f"这里有 {n} 个候选项，每一个是一张图片。\n"
                f"映射关系：{mapping_str}。\n\n"
                f"问题：{question}\n\n"
                f"<think>\n我已经仔细评估了所有候选图片与问题的匹配度，思考完毕。\n</think>\n"
                f"根据以上分析，正确选项只有一个大写字母。正确选项是："
            )
        else:
            text_prompt = (
                f"There are {n} candidate options, each one is an image.\n"
                f"The mapping is: {mapping_str}.\n\n"
                f"Question: {question}\n\n"
                f"<think>\nI have carefully evaluated all candidate images against the question. Thinking process complete.\n</think>\n"
                f"Based on the analysis, the single correct uppercase option letter is: "
            )

        inputs = self._create_inputs_multi(
            text=promote + text_prompt,
            imgs=choice_imgs,
        )

        return inputs, letters

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _base_generation_kwargs(self) -> Dict:
        gen_kwargs: Dict = {
            "do_sample": False,
            # MCQ generates only 1 token, so KV cache is unnecessary.
            # Disabling cache reduces VRAM usage in multi-image scenarios.
            "use_cache": False,
        }

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            if tokenizer.pad_token_id is not None:
                gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
            if tokenizer.eos_token_id is not None:
                gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

        return gen_kwargs

    def generate_choice_only(
        self,
        inputs,
        letters: List[str],
    ) -> Tuple[str, Optional[str]]:
        """
        MCQ-only generation: ForceChoiceTokens constrained decoding, generates exactly 1 token.
        """

        token_to_letter = self._choice_token_map(letters)
        allowed_token_ids = list(token_to_letter.keys())

        gen_kwargs = self._base_generation_kwargs()
        gen_kwargs.update(
            {
                "max_new_tokens": 1,
                "logits_processor": LogitsProcessorList(
                    [ForceChoiceTokens(allowed_token_ids)]
                ),
            }
        )

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            prompt_len = inputs["input_ids"].shape[1]
            new_token_id = int(outputs[0][prompt_len].detach().cpu().item())

        choice = token_to_letter.get(new_token_id)
        raw = choice if choice is not None else self._decode([new_token_id]).strip()

        del outputs
        del inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return raw, choice

    def generate_text_debug(
        self,
        inputs,
        max_new_tokens: int = 64,
    ) -> str:
        """
        For debug use only.
        Do not use this for formal MCQ evaluation — it may emit <think> tokens.
        """

        gen_kwargs = self._base_generation_kwargs()
        gen_kwargs["max_new_tokens"] = max_new_tokens

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

            prompt_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][prompt_len:].detach().cpu().tolist()
            text = self._decode(generated_ids)

        text = _THINK_RE.sub("", text).strip()

        del outputs
        del inputs
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return text

    def generate_text(
        self,
        inputs,
        max_new_tokens: int = 64,
        letters: Optional[List[str]] = None,
    ) -> str:
        """
        Pipeline-compatible entry point (called by VLM_test_parallel.py).

        If letters is provided, uses constrained decoding (generate_choice_only)
        and returns the single letter as a string.
        Otherwise falls back to free-text generation (generate_text_debug).
        """
        if letters is not None:
            raw, _ = self.generate_choice_only(inputs, letters)
            return raw
        return self.generate_text_debug(inputs, max_new_tokens=max_new_tokens)

    # ------------------------------------------------------------------
    # Public evaluation helpers
    # ------------------------------------------------------------------

    def eval_choice_single_img(
        self,
        promote: str,
        img_path: str,
        question: str,
        options: List[str],
        max_new_tokens: int = 1,
        language_type: str = "cn",
    ) -> Tuple[str, Optional[str]]:
        img = self._load_image(img_path)

        inputs = self.choice_single_img_promote(
            promote=promote,
            question=question,
            options=options,
            img=img,
            language_type=language_type,
        )

        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(options)])

        return self.generate_choice_only(
            inputs=inputs,
            letters=letters,
        )

    def eval_choice_multi_img(
        self,
        promote: str,
        img_paths: List[str],
        question: str,
        max_new_tokens: int = 1,
        language_type: str = "cn",
    ) -> Tuple[str, Optional[str]]:
        choice_imgs = [
            self._load_image(path) for path in img_paths
        ]

        inputs, letters = self.choice_multi_img_promote(
            promote=promote,
            question=question,
            choice_imgs=choice_imgs,
            language_type=language_type,
        )

        return self.generate_choice_only(
            inputs=inputs,
            letters=letters,
        )


if __name__ == "__main__":
    model_path = str(config.MODEL_ROOT / "GLM-4.6V-Flash")
    img_root   = str(config.IMAGE_DATA_DIR / "Ancient_Building_Complex_in_the_Wudang_Mountains")

    if not os.path.exists(img_root):
        raise FileNotFoundError(f"Image root does not exist: {img_root}")

    choice_paths = [
        os.path.join(img_root, f)
        for f in sorted(os.listdir(img_root))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ][:4]

    if len(choice_paths) == 0:
        raise RuntimeError(f"No images found in: {img_root}")

    promote = (
        "You are a professional cultural and natural heritage recognition system. "
        "Select the single correct option based on the images, question, and choices. "
        "The final answer must contain only one uppercase option letter. "
        "Do not output explanations, punctuation, tags, or reasoning.\n\n"
    )

    evaluator = GLM46VFlashEvaluator(
        model_path=model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        max_image_side=1024,
        allow_eager_fallback=True,
    )

    # ------------------------------------------------------------
    # Case 1: single image / Chinese
    # ------------------------------------------------------------

    question1_cn = "What architectural style is depicted in the image?"

    options1 = [
        "Modern Skyscraper",
        "Ancient Temple",
        "Beach Resort",
        "Forest Cabin",
    ]

    raw1_cn, choice1_cn = evaluator.eval_choice_single_img(
        promote=promote,
        img_path=choice_paths[0],
        question=question1_cn,
        options=options1,
        max_new_tokens=1,
        language_type="cn",
    )

    print("=== Case 1: Single image / CN ===")
    print("Raw Output:", raw1_cn)
    print("Choice:", choice1_cn)
    print()

    # ------------------------------------------------------------
    # Case 2: single image / English
    # ------------------------------------------------------------

    question1_en = "What architectural style is depicted in the image?"

    raw1_en, choice1_en = evaluator.eval_choice_single_img(
        promote=promote,
        img_path=choice_paths[0],
        question=question1_en,
        options=options1,
        max_new_tokens=1,
        language_type="en",
    )

    print("=== Case 2: Single image / EN ===")
    print("Raw Output:", raw1_en)
    print("Choice:", choice1_en)
    print()

    # ------------------------------------------------------------
    # Case 3: multi image / Chinese
    # ------------------------------------------------------------

    if len(choice_paths) >= 2:
        question2_cn = "Which image best represents Taoist culture?"

        raw2_cn, choice2_cn = evaluator.eval_choice_multi_img(
            promote=promote,
            img_paths=choice_paths,
            question=question2_cn,
            max_new_tokens=1,
            language_type="cn",
        )

        print("=== Case 3: Multi image / CN ===")
        print("Raw Output:", raw2_cn)
        print("Choice:", choice2_cn)
        print()

    # ------------------------------------------------------------
    # Case 4: multi image / English
    # ------------------------------------------------------------

    if len(choice_paths) >= 2:
        question2_en = "Which image best represents Taoist culture?"

        raw2_en, choice2_en = evaluator.eval_choice_multi_img(
            promote=promote,
            img_paths=choice_paths,
            question=question2_en,
            max_new_tokens=1,
            language_type="en",
        )

        print("=== Case 4: Multi image / EN ===")
        print("Raw Output:", raw2_en)
        print("Choice:", choice2_en)
        print()