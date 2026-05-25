import os
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from typing import List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _build_transform(input_size: int = 448) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        tar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - tar)
        if diff < best_diff or (diff == best_diff and area > 0.5 * image_size ** 2 * ratio[0] * ratio[1]):
            best_diff = diff
            best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    w, h = image.size
    aspect_ratio = w / h
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
                for i in range(1, n + 1) for j in range(1, n + 1)
                if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    best = _find_closest_aspect_ratio(aspect_ratio, target_ratios, w, h, image_size)
    target_w, target_h = image_size * best[0], image_size * best[1]
    blocks = best[0] * best[1]
    resized = image.resize((target_w, target_h))
    tiles = [
        resized.crop((
            (i % best[0]) * image_size,
            (i // best[0]) * image_size,
            ((i % best[0]) + 1) * image_size,
            ((i // best[0]) + 1) * image_size,
        ))
        for i in range(blocks)
    ]
    if use_thumbnail and blocks != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def _load_image_tensor(
    img: Image.Image,
    input_size: int = 448,
    max_num: int = 12,
) -> torch.Tensor:
    transform = _build_transform(input_size)
    tiles = _dynamic_preprocess(img, image_size=input_size, max_num=max_num, use_thumbnail=True)
    return torch.stack([transform(t) for t in tiles])  # (num_tiles, C, H, W)


class InternVL25Evaluator:
    """
    Evaluator for InternVL2.5-8B.

    Loading: AutoModel + AutoTokenizer (trust_remote_code=True)
    Inference: model.chat() official high-level API
    Multi-image: native support via <image> placeholder + num_patches_list
    """

    def __init__(
        self,
        model_path: str,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        max_num_tiles: int = 12,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, use_fast=False
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        ).eval()
        self.dtype = dtype
        self.max_num_tiles = max_num_tiles
        self.device = next(self.model.parameters()).device

    def _img_to_tensor(self, img: Image.Image) -> torch.Tensor:
        return _load_image_tensor(img, max_num=self.max_num_tiles).to(self.dtype).to(self.device)

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def choice_single_img_promote(
        self,
        promote: str,
        question: str,
        options: List[str],
        img: Image.Image,
        language_type: str = "cn",
    ) -> Tuple[torch.Tensor, str, None]:
        n = len(options)
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

        pixel_values = self._img_to_tensor(img)
        query = f"<image>\n{promote}{text_prompt}"
        return pixel_values, query, None  # num_patches_list=None for single image

    def choice_multi_img_promote(
        self,
        promote: str,
        question: str,
        choice_imgs: List[Image.Image],
        language_type: str = "cn",
    ) -> Tuple[torch.Tensor, str, List[int], List[str]]:
        n = len(choice_imgs)
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

        tensors = [self._img_to_tensor(img) for img in choice_imgs]
        num_patches_list = [t.size(0) for t in tensors]
        pixel_values = torch.cat(tensors, dim=0)

        img_tags = "".join(f"Image-{i+1}: <image>\n" for i in range(n))
        query = f"{img_tags}{promote}{text_prompt}"
        return pixel_values, query, num_patches_list, letters

    # ------------------------------------------------------------------
    # Generation (via model.chat)
    # ------------------------------------------------------------------

    def _chat(
        self,
        pixel_values: torch.Tensor,
        query: str,
        max_new_tokens: int,
        num_patches_list=None,
    ) -> str:
        gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=False)
        kwargs = {}
        if num_patches_list is not None:
            kwargs["num_patches_list"] = num_patches_list
        response = self.model.chat(
            self.tokenizer, pixel_values, query, gen_cfg, **kwargs
        )
        torch.cuda.empty_cache()
        return response.strip()

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
        img = Image.open(img_path).convert("RGB")
        pixel_values, query, _ = self.choice_single_img_promote(
            promote, question, options, img, language_type
        )
        out = self._chat(pixel_values, query, max_new_tokens)
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
        choice_imgs = [Image.open(p).convert("RGB") for p in img_paths]
        pixel_values, query, num_patches_list, letters = self.choice_multi_img_promote(
            promote, question, choice_imgs, language_type
        )
        out = self._chat(pixel_values, query, max_new_tokens, num_patches_list)
        pattern = r"(?<![A-Za-z])([" + "".join(letters) + r"])(?=[^A-Za-z]|$)"
        m = re.search(pattern, out)
        return out, m.group(1) if m else None


if __name__ == "__main__":
    model_path = str(config.MODEL_ROOT / "InternVL2_5-8B")
    img_data_root = str(config.IMAGE_DATA_DIR)

    def _first_img(site: str) -> str:
        d = os.path.join(img_data_root, site)
        imgs = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        return os.path.join(d, imgs[0])

    evaluator = InternVL25Evaluator(model_path)
    promote = (
        "You are a professional cultural and natural heritage recognition system. "
        "Please carefully examine the image and read through options A, B, C, D, and E, "
        "then identify the single correct answer based on the visual content.\n"
        "Requirements:\n- Output only the letter of the correct option (A / B / C / D / E)\n"
        "- Do not include any explanation, punctuation, or additional content\nExample answer: A\n"
    )

    img_single = _first_img("Ancient_Building_Complex_in_the_Wudang_Mountains")

    raw1, choice1 = evaluator.eval_choice_single_img(
        promote, img_single, "What type of cultural heritage site is shown in the image?",
        ["Royal garden", "Taoist holy site", "Buddhist temple", "Confucian academy"],
        max_new_tokens=16, language_type="cn",
    )
    print("=== Case 1: Single image / CN ===")
    print("raw   :", raw1)
    print("choice:", choice1, " (expected: B)\n")

    raw1e, choice1e = evaluator.eval_choice_single_img(
        promote, img_single, "What type of cultural heritage site is shown in the image?",
        ["Royal garden", "Taoist holy site", "Buddhist temple", "Confucian academy"],
        max_new_tokens=8, language_type="en",
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
        promote, multi_imgs, "Which image shows a rice terrace landscape?",
        max_new_tokens=16, language_type="cn",
    )
    print("=== Case 2: Multi image / CN ===")
    print("raw   :", raw2)
    print("choice:", choice2, " (expected: D)\n")
