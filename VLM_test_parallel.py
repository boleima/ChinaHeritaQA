"""
VLM_test_parallel.py — Producer-consumer pipeline for VLM evaluation.

Key optimisation:
  PIL image loading is offloaded to a background ThreadPoolExecutor so it
  overlaps with GPU inference on the main thread (pipeline / prefetch mode).
  This hides per-sample I/O latency (~10-200 ms) behind GPU inference (~1-10 s).

  Pipeline mode  — evaluator exposes choice_*_promote(PIL) + generate_text():
      Qwen2-VL, Qwen2.5-VL, Qwen3-VL, GLM-4.6V, CogVLM2, Pixtral, Llama-3.2

  Sequential mode — evaluator bundles I/O + GPU prep internally:
      deepseek-vl, deepseek-vl2, InternVL2.5

Extra flags:
  --prefetch N   depth of the prefetch queue (default 4)
  --skip_done    skip output files that already exist (resume interrupted run)
"""
import os
# Prevents CUDA allocator fragmentation over many inferences with variable-size inputs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import re
import config
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm
import argparse

from utils.inference import dataset_load, is_correct, save_as_

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Strings that mark the start of a hallucinated follow-up turn.
_CONV_STOP = ("\nUser:", "\nuser:", "\n用户:", "\nAssistant:", "\nassistant:")
model_id_list = [
    "Qwen3-VL-8B-Instruct",           # 0
    "GLM-4.6V-Flash",                 # 1
    "InternVL2_5-8B",                 # 2
    "deepseek-vl2-small",             # 3
    "Qwen2.5-VL-7B-Instruct",         # 4
    "CogVLM2-19B",                    # 5
    "Qwen2-VL-14B-Instruct",          # 6  (reserved – no evaluator yet)
    "Pixtral-12B-2409",               # 7  (reserved – no evaluator yet)
    "Llama-3.2-11B-Vision-Instruct",  # 8  (reserved – no evaluator yet)
]
ALL_QUESTIONS = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7']
language_list = ['cn', 'en']

_SENTINEL = object()

# Models whose choice_*_promote() accept PIL.Image and expose generate_text().
# InternVL25 uses a different internal API (model.chat) → sequential only.
# DeepSeek variants bundle path loading internally → sequential only.
_PIPELINE_KEYS = ('Qwen', 'GLM', 'CogVLM', 'Pixtral', 'Llama')


# ---------------------------------------------------------------------------
# Evaluator factory
# ---------------------------------------------------------------------------

def build_evaluator(rp, model_type):
    path = os.path.join(rp, model_type)
    if 'Qwen3' in model_type:
        from Vlm_Eval.eval_qwen3 import Qwen3VLEvaluator
        return Qwen3VLEvaluator(path)
    elif 'Qwen2.5' in model_type:
        from Vlm_Eval.eval_qwen25 import Qwen25VLEvaluator
        return Qwen25VLEvaluator(path)
    elif 'Qwen' in model_type:
        from Vlm_Eval.eval_qwen2 import QwenVLEvaluator
        return QwenVLEvaluator(path)
    elif 'deepseek-vl2' in model_type:
        from Vlm_Eval.eval_deepseek_vl2 import DeepSeekVL2Evaluator
        return DeepSeekVL2Evaluator(path)
    elif 'GLM-4.6' in model_type:
        from Vlm_Eval.eval_Glm46v import GLM46VFlashEvaluator
        return GLM46VFlashEvaluator(path, max_image_side=512)
    elif 'CogVLM' in model_type:
        from Vlm_Eval.eval_Cogvlm import CogVLM2Evaluator
        return CogVLM2Evaluator(path)
    elif 'InternVL' in model_type:
        from Vlm_Eval.eval_internvl25 import InternVL25Evaluator
        return InternVL25Evaluator(path)
    elif 'Pixtral' in model_type:
        from Vlm_Eval.eval_pixtral import PixtralEvaluator
        return PixtralEvaluator(path)
    elif 'Llama' in model_type:
        from Vlm_Eval.eval_llama32v import Llama32VisionEvaluator
        return Llama32VisionEvaluator(path)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _supports_pipeline(model_type):
    return any(k in model_type for k in _PIPELINE_KEYS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pil(path: str) -> Image.Image:
    return Image.open(path).convert('RGB')


def _extract_letter(pred_text: str, n_options: int):
    """Return the first valid option letter found in model output, or None."""
    if not pred_text:
        return None
    text = pred_text
    for stop in _CONV_STOP:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    letters = LETTERS[:n_options]
    m = re.search(r"(?<![A-Za-z])([" + letters + r"])(?=[^A-Za-z]|$)", text)
    return m.group(1) if m else None


def _make_row(index, images, q_text, options, gold_letters,
              pred_text, pred_letter, answer_state):
    row = {
        "index": index,
        "image": images if isinstance(images, str) else None,
        "question": q_text,
        "gold": ",".join(gold_letters),
        "pred": ",".join(pred_letter) if pred_letter else "",
        "correct": int(answer_state),
        "raw_output": pred_text,
    }
    cols = options if isinstance(images, str) else images
    for i, val in enumerate(cols):
        row[f"options_{LETTERS[i]}"] = val
    return row


# ---------------------------------------------------------------------------
# Producer thread: parallel image loading (pure I/O, no GPU, thread-safe)
# ---------------------------------------------------------------------------

def _image_loader_thread(items, out_q: queue.Queue):
    """
    Load PIL images in a background thread and push to out_q.
    The out_q has a maxsize, so out_q.put() will block when full.
    This creates perfect back-pressure, ensuring we only ever hold 
    `maxsize` images in RAM at any given time, preventing OOM.
    """
    def _load_item(item):
        _, _, images, _, _, _ = item
        if isinstance(images, str):
            return _load_pil(images)
        return [_load_pil(p) for p in images]

    for item in items:
        try:
            pil = _load_item(item)
            out_q.put((item, pil, None))  # blocks when queue is full (default 4), creating back-pressure
        except Exception as exc:
            out_q.put((item, None, exc))

    # signal end of stream
    out_q.put(_SENTINEL)


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------

def evaluate_pipeline(evaluator, items, lan: str, prefetch_size: int = 4):
    """
    Pipeline mode (Qwen / GLM / Yi / Idefics2 / CogVLM).

    Timeline for item N:
      [producer thread] load image N+k  ─────────────┐
      [main thread]     tokenize N → GPU infer N      │ overlap
    """
    q = queue.Queue(maxsize=prefetch_size)
    loader = threading.Thread(
        target=_image_loader_thread, args=(items, q), daemon=True
    )
    loader.start()

    rows = []
    total = correct = 0

    with tqdm(total=len(items), desc="Evaluating [pipeline]", unit="item") as pbar:
        while True:
            payload = q.get()
            if payload is _SENTINEL:
                break

            item, pil, exc = payload
            index, promote, images, q_text, options, gold_letters = item

            if exc is not None:
                print(f"\n[Error] image load index={index}: {exc}")
                pbar.update(1)
                continue

            max_tok = 32
            _use_constrained = hasattr(evaluator, 'generate_choice_only')
            try:
                if isinstance(images, str):
                    inputs = evaluator.choice_single_img_promote(
                        promote, q_text, options, pil, lan
                    )
                    if _use_constrained:
                        choice_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(options)])
                        pred_text = evaluator.generate_text(
                            inputs, max_new_tokens=max_tok, letters=choice_letters
                        )
                    else:
                        pred_text = evaluator.generate_text(inputs, max_new_tokens=max_tok)
                    pred_letter = _extract_letter(pred_text, len(options))
                else:
                    inputs, choice_letters = evaluator.choice_multi_img_promote(
                        promote, q_text, pil, lan
                    )
                    if _use_constrained:
                        pred_text = evaluator.generate_text(
                            inputs, max_new_tokens=max_tok, letters=choice_letters
                        )
                    else:
                        pred_text = evaluator.generate_text(inputs, max_new_tokens=max_tok)
                    pred_letter = _extract_letter(pred_text, len(images))
            except Exception as exc:
                print(f"\n[Error] inference index={index}: {exc}")
                pred_text, pred_letter = "", None

            answer_state = is_correct(pred_letter, gold_letters)
            total += 1
            correct += int(answer_state)
            rows.append(_make_row(
                index, images, q_text, options, gold_letters,
                pred_text, pred_letter, answer_state,
            ))
            pbar.update(1)

    loader.join()
    return rows, total, correct


def evaluate_sequential(evaluator, items, lan: str):
    """Sequential fallback (DeepSeek: bundles I/O + GPU prep internally)."""
    rows = []
    total = correct = 0

    max_tok = 32
    for item in tqdm(items, desc="Evaluating [sequential]", unit="item"):
        index, promote, images, q_text, options, gold_letters = item
        try:
            if isinstance(images, str):
                pred_text, pred_letter = evaluator.eval_choice_single_img(
                    promote=promote, img_path=images, question=q_text,
                    options=options, max_new_tokens=max_tok, language_type=lan,
                )
            else:
                pred_text, pred_letter = evaluator.eval_choice_multi_img(
                    promote=promote, img_paths=images, question=q_text,
                    max_new_tokens=max_tok, language_type=lan,
                )
        except Exception as exc:
            print(f"\n[Error] inference index={index}: {exc}")
            pred_text, pred_letter = "", None

        answer_state = is_correct(pred_letter, gold_letters)
        total += 1
        correct += int(answer_state)
        rows.append(_make_row(
            index, images, q_text, options, gold_letters,
            pred_text, pred_letter, answer_state,
        ))

    return rows, total, correct


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM parallel inference pipeline")
    parser.add_argument("--model_id", default=0,
                        help="0:Qwen3-VL, 1:GLM-4.6V, 2:InternVL2.5, 3:deepseek-vl2, "
                             "4:Qwen2.5-VL, 5:CogVLM2, 6:Qwen2-VL-14B(reserved), "
                             "7:Pixtral(reserved), 8:Llama-3.2(reserved)")
    parser.add_argument("--prefetch", type=int, default=4,
                        help="Prefetch queue depth for pipeline mode (default: 4)")
    parser.add_argument("--skip_done", action="store_true",
                        help="Skip output files that already exist")
    parser.add_argument("--lang", nargs="+", choices=language_list, default=language_list,
                        help="Language(s) to evaluate: cn en (default: both)")
    parser.add_argument("--questions", nargs="*", default=ALL_QUESTIONS,
                        help="Question sets to evaluate, e.g. q1 q3 q5 (default: all)")
    args = parser.parse_args()

    rp           = str(config.MODEL_ROOT)
    model_type   = model_id_list[int(args.model_id)]
    dataset_root = str(config.QUESTION_DIR)
    image_root   = str(config.IMAGE_ROOT)
    output_rp    = str(config.RESULTS_DIR / model_type)
    os.makedirs(output_rp, exist_ok=True)

    pipeline_mode = _supports_pipeline(model_type)
    q_set = args.questions if args.questions else ALL_QUESTIONS
    print(f"Model: {model_type}  |  mode: {'pipeline' if pipeline_mode else 'sequential'}")
    print(f"q_set: {q_set}  |  lang: {args.lang}")

    evaluator = build_evaluator(rp, model_type)
    for question_dir in os.listdir(dataset_root):
        if question_dir.split('.')[0] not in q_set:
            continue
        dataset_path = os.path.join(dataset_root, question_dir)
        for lan in args.lang:
            print(f"\nEvaluating question_dir={question_dir} language={lan} ...")
            q_name = question_dir.split('.')[0]
            output_file = os.path.join(output_rp, f"Evaluator_{q_name}_{lan}.xlsx")

            if args.skip_done and os.path.exists(output_file):
                print(f"[Skip] {output_file}")
                continue

            items = list(dataset_load(dataset_path, image_root, language=lan))

            if pipeline_mode:
                rows, total, correct = evaluate_pipeline(
                    evaluator, items, lan, prefetch_size=args.prefetch
                )
            else:
                rows, total, correct = evaluate_sequential(evaluator, items, lan)

            acc = correct / total if total else 0.0
            save_as_(rows, output_file)
            print(f"Total: {total} | Correct: {correct} | Accuracy: {acc:.4f} | Output: {output_file}")
