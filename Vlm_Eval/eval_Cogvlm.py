import os
import re
import sys
import math
import types
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Bootstrap: add project root to sys.path before any local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==========================================
# [Config] Prevent "Disk quota exceeded" errors
# ==========================================
os.makedirs(str(config.TRITON_CACHE), exist_ok=True)
os.makedirs(str(config.HF_CACHE), exist_ok=True)

os.environ["TRITON_CACHE_DIR"] = str(config.TRITON_CACHE)
os.environ["HF_HOME"]          = str(config.HF_CACHE)


class CogVLM2Evaluator:
    def __init__(
        self,
        model_path: str,
        dtype=None,
        device_map: str = "cuda",
        trust_remote_code: bool = True,
    ):
        self.device = device_map if device_map != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if dtype is None:
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.dtype = dtype

        print(f"Loading CogVLM2 from {model_path} with {self.dtype} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            fix_mistral_regex=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            trust_remote_code=trust_remote_code,
        ).to(self.device).eval()

        # [Patch] compatibility fix for transformers >= 4.44
        if not hasattr(self.model, "_extract_past_from_model_output"):
            def _extract_past_from_model_output_patch(self, outputs, standardize_cache_format=False):
                return "past_key_values", outputs.get("past_key_values", None)

            self.model._extract_past_from_model_output = types.MethodType(
                _extract_past_from_model_output_patch,
                self.model,
            )

        # [Patch] DynamicCache -> legacy tuple cache
        try:
            from transformers.cache_utils import DynamicCache

            _orig_llm_forward = self.model.model.llm_forward

            def _llm_forward_patched(*args, **kwargs):
                if "past_key_values" in kwargs and isinstance(kwargs["past_key_values"], DynamicCache):
                    kwargs["past_key_values"] = kwargs["past_key_values"].to_legacy_cache()
                elif len(args) > 3 and isinstance(args[3], DynamicCache):
                    args = args[:3] + (args[3].to_legacy_cache(),) + args[4:]
                return _orig_llm_forward(*args, **kwargs)

            self.model.model.llm_forward = _llm_forward_patched

        except (ImportError, AttributeError):
            pass

    # ==========================================================
    # Input preparation and generation
    # ==========================================================

    def _prepare_inputs(
        self,
        query: str,
        images: Optional[List[Image.Image]],
    ) -> Dict[str, torch.Tensor]:
        image_list = images if images is not None else []

        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=query,
            history=[],
            images=image_list,
            template_version="chat",
        )

        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(self.device),
            "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(self.device),
            "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(self.device),
        }

        if image_list:
            inputs["images"] = [
                [
                    img.to(self.device).to(self.dtype)
                    for img in input_by_model["images"]
                ]
            ]

        return inputs

    def generate_text(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int = 16,
    ) -> str:
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.model.config.eos_token_id

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": 1,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "use_cache": True,
            "do_sample": False,
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        response = response.replace("<|end_of_text|>", "")
        return response.strip()

    # ==========================================================
    # Answer parsing
    # ==========================================================

    @staticmethod
    def parse_choice(out: str, letters: List[str]) -> Optional[str]:
        if not out:
            return None

        valid = "".join(letters)
        text = out.strip().upper()

        # Common answer formats: answer is A / Answer: A / Option A
        patterns = [
            rf"(?:答案|选项)\s*(?:是|为)?\s*[:：]?\s*([{valid}])",  # Chinese: "答案是A"
            rf"(?:ANSWER|OPTION|CHOICE)\s*(?:IS)?\s*[:：]?\s*([{valid}])",
            rf"^\s*([{valid}])(?:\s|[,.，。:：、)\]]|$)",
        ]

        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                return m.group(1)

        # Fallback: match standalone letter, avoiding picking 'A' from within 'ANSWER'
        m = re.search(rf"(?<![A-Z])([{valid}])(?![A-Z])", text)
        return m.group(1) if m else None

    # ==========================================================
    # Single-image + text-option QA
    # ==========================================================

    def choice_single_img_prompt(
        self,
        promote: str,
        question: str,
        options: List[str],
        img: Image.Image,
        language_type: str = "cn",
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        n = len(options)
        if n < 1 or n > 26:
            raise ValueError("Number of options must be between 1 and 26.")

        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n])
        options_text = "\n".join([f"{letters[i]}. {options[i]}" for i in range(n)])

        prefix = promote.strip() + "\n" if promote else ""

        if language_type == "cn":
            text_prompt = (
                f"{prefix}"
                f"这是一道单图选择题。请根据图片和问题，从选项中选择最合适的一项。\n"
                f"问题：{question}\n\n"
                f"选项：\n{options_text}\n\n"
                f"重要：请只输出一个大写字母，例如 A。不要输出解释、标点或其他文字。"
            )
        else:
            text_prompt = (
                f"{prefix}"
                f"This is a single-image multiple-choice question. "
                f"Choose the best option based on the image and the question.\n"
                f"Question: {question}\n\n"
                f"Options:\n{options_text}\n\n"
                f"Important: output only one uppercase letter, for example A. "
                f"Do not output explanations, punctuation, or any other text."
            )

        inputs = self._prepare_inputs(query=text_prompt, images=[img])
        return inputs, letters

    def eval_choice_single_img(
        self,
        promote: str,
        img_path: str,
        question: str,
        options: List[str],
        max_new_tokens: int = 16,
        language_type: str = "cn",
    ) -> Tuple[str, Optional[str]]:
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        with Image.open(img_path) as im:
            img = ImageOps.exif_transpose(im).convert("RGB").copy()

        inputs, letters = self.choice_single_img_prompt(
            promote=promote,
            question=question,
            options=options,
            img=img,
            language_type=language_type,
        )

        out = self.generate_text(inputs, max_new_tokens=max_new_tokens)
        choice = self.parse_choice(out, letters)
        return out, choice

    # ==========================================================
    # Multi-image option QA
    # ==========================================================

    @staticmethod
    def _load_font(font_size: int = 46):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            "DejaVuSans-Bold.ttf",
        ]

        for path in candidates:
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue

        return ImageFont.load_default()

    @staticmethod
    def _make_labeled_choice_grid(
        choice_imgs: List[Image.Image],
        letters: List[str],
        image_size: int = 384,
        padding: int = 24,
        label_h: int = 64,
    ) -> Image.Image:
        n = len(choice_imgs)
        if n < 1:
            raise ValueError("choice_imgs must not be empty.")

        # A-D: 2 columns; A-E or more: 3 columns to avoid excessive width
        cols = 2 if n <= 4 else 3
        rows = math.ceil(n / cols)

        cell_w = image_size + padding * 2
        cell_h = image_size + label_h + padding * 2

        canvas_w = cols * cell_w
        canvas_h = rows * cell_h
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

        draw = ImageDraw.Draw(canvas)
        font = CogVLM2Evaluator._load_font(font_size=46)

        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS

        for i, raw_img in enumerate(choice_imgs):
            letter = letters[i]

            col = i % cols
            row = i // cols

            cell_x0 = col * cell_w
            cell_y0 = row * cell_h
            cell_x1 = cell_x0 + cell_w - 1
            cell_y1 = cell_y0 + cell_h - 1

            # cell border
            draw.rectangle(
                [cell_x0 + 8, cell_y0 + 8, cell_x1 - 8, cell_y1 - 8],
                outline="black",
                width=4,
            )

            # top label bar
            label_box = [
                cell_x0 + 8,
                cell_y0 + 8,
                cell_x1 - 8,
                cell_y0 + label_h,
            ]
            draw.rectangle(label_box, fill="black")

            label_text = letter
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            text_x = cell_x0 + (cell_w - text_w) // 2
            text_y = cell_y0 + 8 + (label_h - text_h) // 2 - 4
            draw.text((text_x, text_y), label_text, fill="white", font=font)

            # process image
            img = ImageOps.exif_transpose(raw_img).convert("RGB")
            img.thumbnail((image_size, image_size), resample=resample)

            img_x = cell_x0 + padding + (image_size - img.width) // 2
            img_y = cell_y0 + padding + label_h + (image_size - img.height) // 2

            canvas.paste(img, (img_x, img_y))

        return canvas

    def choice_multi_img_prompt(
        self,
        promote: str,
        question: str,
        choice_imgs: List[Image.Image],
        language_type: str = "cn",
        debug_grid_path: Optional[str] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        n = len(choice_imgs)
        if n < 1 or n > 26:
            raise ValueError("Number of choice images must be between 1 and 26.")

        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n])

        # CogVLM2's standard image/chat model has unstable support for multiple
        # independent images; stitch all candidate images into a single labeled grid.
        grid_img = self._make_labeled_choice_grid(choice_imgs, letters)

        if debug_grid_path is not None:
            os.makedirs(os.path.dirname(debug_grid_path) or ".", exist_ok=True)
            grid_img.save(debug_grid_path)

        option_text = " / ".join(letters)
        prefix = promote.strip() + "\n" if promote else ""

        if language_type == "cn":
            text_prompt = (
                f"{prefix}"
                f"这是一道多图选择题。图片中每个候选项已经用大写字母标注为 {option_text}。\n"
                f"请根据图片内容回答问题。\n\n"
                f"问题：{question}\n\n"
                f"重要：请只输出一个大写字母，例如 A。"
                f"不要输出解释、标点或其他文字。"
                f"即使前面的指令要求解释，也必须只输出一个选项字母。"
            )
        else:
            text_prompt = (
                f"{prefix}"
                f"This is a multi-image multiple-choice question. "
                f"Each candidate image is labeled directly in the image as {option_text}.\n"
                f"Answer the question based on the image content.\n\n"
                f"Question: {question}\n\n"
                f"Important: output only one uppercase letter, for example A. "
                f"Do not output explanations, punctuation, or any other text. "
                f"Even if previous instructions ask for explanation, output only one option letter."
            )

        inputs = self._prepare_inputs(query=text_prompt, images=[grid_img])
        return inputs, letters

    def eval_choice_multi_img(
        self,
        promote: str,
        img_paths: List[str],
        question: str,
        max_new_tokens: int = 16,
        language_type: str = "cn",
        debug_grid_path: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        if not img_paths:
            raise ValueError("img_paths must not be empty.")

        choice_imgs = []
        for path in img_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(path)

            with Image.open(path) as im:
                choice_imgs.append(ImageOps.exif_transpose(im).convert("RGB").copy())

        inputs, letters = self.choice_multi_img_prompt(
            promote=promote,
            question=question,
            choice_imgs=choice_imgs,
            language_type=language_type,
            debug_grid_path=debug_grid_path,
        )

        out = self.generate_text(inputs, max_new_tokens=max_new_tokens)
        choice = self.parse_choice(out, letters)
        return out, choice


if __name__ == "__main__":
    MODEL_PATH = str(config.MODEL_ROOT / "CogVLM2-19B")
    img_root   = str(config.IMAGE_DATA_DIR / "Ancient_Building_Complex_in_the_Wudang_Mountains")

    if os.path.exists(img_root):
        evaluator = CogVLM2Evaluator(MODEL_PATH)

        files = [
            f for f in os.listdir(img_root)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        files = sorted(files)

        choice_paths = [os.path.join(img_root, f) for f in files[:5]]

        # No explanation required — prevents empty or overly long outputs that fail parsing
        promote = (
            "You are a multiple-choice QA system. "
            "Select the best answer from the given options. "
            "The final answer must contain only one uppercase letter."
        )

        # Test 1: single image + text options
        if len(choice_paths) > 0:
            print("\n--- Single Image Test ---")

            q1 = "What architectural style is shown?"
            opts1 = ["Modern", "Ancient Chinese", "Gothic", "Industrial", "Baroque"]

            raw1, c1 = evaluator.eval_choice_single_img(
                promote=promote,
                img_path=choice_paths[0],
                question=q1,
                options=opts1,
                max_new_tokens=16,
                language_type="en",
            )
            print(f"RAW repr: {repr(raw1)}")
            print(f"Choice: {c1}")

            raw2, c2 = evaluator.eval_choice_single_img(
                promote=promote,
                img_path=choice_paths[0],
                question="What architectural style is shown in the image?",
                options=opts1,
                max_new_tokens=16,
                language_type="cn",
            )
            print(f"RAW repr: {repr(raw2)}")
            print(f"Choice: {c2}")

        # Test 2: multi-image options
        if len(choice_paths) >= 2:
            print("\n--- Multi Image Choice Test ---")

            raw3, c3 = evaluator.eval_choice_multi_img(
                promote=promote,
                img_paths=choice_paths[:5],
                question="Which option shows Asian-style architecture?",
                max_new_tokens=16,
                language_type="cn",
                debug_grid_path="./debug_choice_grid_cn.jpg",
            )
            print(f"RAW repr: {repr(raw3)}")
            print(f"Choice: {c3}")
            print("Debug grid saved to: ./debug_choice_grid_cn.jpg")

            raw4, c4 = evaluator.eval_choice_multi_img(
                promote=promote,
                img_paths=choice_paths[:5],
                question="Which option is Asian-style architecture?",
                max_new_tokens=16,
                language_type="en",
                debug_grid_path="./debug_choice_grid_en.jpg",
            )
            print(f"RAW repr: {repr(raw4)}")
            print(f"Choice: {c4}")
            print("Debug grid saved to: ./debug_choice_grid_en.jpg")