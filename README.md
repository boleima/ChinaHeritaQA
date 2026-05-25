# ChinaHeritaQA

**ChinaHeritaQA** is a bilingual (Chinese / English) visual question answering benchmark for evaluating vision-language models' (VLMs) ability to recognise and understand **Chinese UNESCO World Heritage Sites**. The dataset covers 51 Chinese heritage sites and 23 non-Chinese sites as distractors, with images sourced from real social-media posts (Weibo).

---

## Overview

| Item | Details |
|------|---------|
| Chinese heritage sites | 58 |
| Non-Chinese sites (distractors) | 23 |
| Question types | 7 (q1 – q7) |
| Questions per type | 1,370 – 2,279 |
| Evaluation languages | Chinese (`cn`) · English (`en`) |
| Answer format | 5-choice single-select (A / B / C / D / E) |

### Sample Images

<table>
  <tr>
    <td align="center"><img src="DATA/Images/Image_data/Ancient_Building_Complex_in_the_Wudang_Mountains/5206045443491227_7.jpg" width="220"/><br><sub>Ancient Building Complex in the Wudang Mountains</sub></td>
    <td align="center"><img src="DATA/Images/Image_data/Ancient_Building_Complex_in_the_Wudang_Mountains/5118017971685704_7.jpg" width="220"/><br><sub>Ancient Building Complex in the Wudang Mountains</sub></td>
    <td align="center"><img src="DATA/Images/worlds_data/Archaeological_Areas_of_Pompei/3.jpg" width="220"/><br><sub>Archaeological Areas of Pompei (non-Chinese distractor)</sub></td>
  </tr>
</table>

---

## Question Types

| ID | Script | Task | Option type | Count |
|----|--------|------|-------------|------:|
| **q1** | `q1_location_recognition.py` | Identify which heritage site is shown in the image | 5 text options (site names) | 2,279 |
| **q2** | `q2_image_retrieval.py` | Given a site name, select the matching image from 5 candidates | 5 image options | 2,279 |
| **q3** | `q3_intro_matching.py` | Select the correct brief description for the image | 5 text options (descriptions) | 2,279 |
| **q4** | `q4_dynasty_classification.py` | Identify the dynasty in which the architectural complex was built | 5 text options (dynasty names) | 1,658 |
| **q5** | `q5_historical_background.py` | Select the correct historical background description for the image | 5 text options | 1,989 |
| **q6** | `q6_main_function.py` | Select the correct main-function description for the image | 5 text options | 2,279 |
| **q7** | `q7_architectural_usage.py` | Select the correct architectural-usage description for the image | 5 text options | 1,370 |

### Distractor Strategy (q1, q2, q5, q6, q7)

```
Option A  Correct answer
Option B  Same heritage type  (e.g. both classified as "ancient city")
Option C  Same province as the correct site
Option D  Any other Chinese heritage site
Option E  Non-Chinese World Heritage Site
```

---

## Repository Layout

```
ChinaHeritaQA/
├── config.py                          # ← Edit only this file (MODEL_ROOT and BIG_DISK_ROOT)
│
├── DATA/
│   ├── heritage_meta_weibo_V1.json    # Metadata for 51 Chinese heritage sites
│   ├── world_heritage_info_V1.json    # Metadata for 23 non-Chinese heritage sites
│   ├── heritage_city.json             # Heritage sites indexed by province
│   ├── heritage_type.json             # Heritage sites indexed by type
│   ├── dynast_list_V1.json            # Chinese dynasties and European historical eras
│   ├── heritage_brief_intro.json      # Brief descriptions (used by q3)
│   ├── quesion_info/                  # Generated question JSONs (q1.json … q7.json)
│   ├── question_results/              # Evaluation outputs (one .xlsx per model × type × language)
│   └── Images/                        # sample Images
│       ├── Image_data/<site_name>/    # Chinese heritage site images (Weibo)
│       └── worlds_data/<site_name>/   # Non-Chinese heritage site images
│
├── Question_Gen/
│   ├── generate_questions.py          # Entry script to batch-generate all question types
│   ├── q1_location_recognition.py
│   ├── q2_image_retrieval.py
│   ├── q3_intro_matching.py
│   ├── q4_dynasty_classification.py
│   ├── q5_historical_background.py
│   ├── q6_main_function.py
│   └── q7_architectural_usage.py
│
├── Vlm_Eval/
│   ├── eval_qwen3.py                  # Qwen3-VL-8B-Instruct
│   ├── eval_qwen25.py                 # Qwen2.5-VL-7B-Instruct
│   ├── eval_Glm46v.py                 # GLM-4.6V-Flash (constrained decoding)
│   ├── eval_internvl25.py             # InternVL2.5-8B
│   ├── eval_deepseek_vl2.py           # DeepSeek-VL2-Small
│   └── eval_Cogvlm.py                 # CogVLM2-19B
│
├── utils/
│   ├── inference.py                   # dataset_load · is_correct · save_as_
│   └── data_preprocess.py             # Option builder, province parser, JSON helpers
│
└── VLM_test_parallel.py               # Main evaluation entry (pipeline / sequential mode)
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd ChinaHeritaQA
pip install torch torchvision transformers accelerate tqdm pandas openpyxl pillow
# Required for Qwen-VL series (recommended):
pip install qwen-vl-utils
```

### 2. Download model weights

Download model weights to a single directory, e.g. `/data/models`:

```
/data/models/
├── Qwen3-VL-8B-Instruct/
├── Qwen2.5-VL-7B-Instruct/
├── GLM-4.6V-Flash/
├── InternVL2_5-8B/
├── deepseek-vl2-small/
└── CogVLM2-19B/
```

### 3. Prepare image data

Place heritage site images under `DATA/Images/`:

```
DATA/Images/
├── Image_data/
│   ├── Ancient_Building_Complex_in_the_Wudang_Mountains/
│   │   ├── 5119769210784553_1.jpg
│   │   └── ...
│   └── <other site names>/
└── worlds_data/
    ├── Archaeological_Areas_of_Pompei/
    │   ├── 1.jpg
    │   └── ...
    └── <other site names>/
```

After adding images, run the following snippet once to sync image paths in the JSON metadata:

```python
from utils.data_preprocess import update_image_urls
import config
update_image_urls(str(config.HERITAGE_META), str(config.IMAGE_DATA_DIR))
```

### 4. Edit `config.py`

Open [config.py](config.py) and set the two required paths:

```python
# Root directory containing all downloaded model weight folders.
# Each subfolder name must match an entry in model_id_list of VLM_test_parallel.py.
MODEL_ROOT    = Path("/data/models")       # ← set this

# Root directory on a large-capacity disk for Triton / HuggingFace caches (≥ 50 GB recommended).
BIG_DISK_ROOT = Path("/scratch/username")  # ← set this (HPC environment)
```

All other paths are derived automatically from `ROOT` (the project directory).

---

## Generating Questions

Generate all 7 question types in sequence:

```bash
python Question_Gen/generate_questions.py
```

Output files are saved to `DATA/quesion_info/q1.json` … `q7.json`.

To regenerate only specific question types, comment out the corresponding entries in the `q_list` inside `generate_questions.py`.

---

## Running Evaluation

```bash
python VLM_test_parallel.py \
    --model_id 0 \
    --lang cn en \
    --questions q1 q2 q3 q4 q5 q6 q7 \
    --prefetch 4 \
    --skip_done
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_id` | `0` | Model index (see table below) |
| `--lang` | `cn en` | Evaluation language(s): `cn`, `en`, or both |
| `--questions` | all | Question types to evaluate, e.g. `q1 q5` |
| `--prefetch` | `4` | Prefetch queue depth for pipeline mode |
| `--skip_done` | off | Skip existing output files (supports resume) |

### Model Index

| ID | Model name | Evaluator file |
|----|-----------|----------------|
| 0 | Qwen3-VL-8B-Instruct | `eval_qwen3.py` |
| 1 | GLM-4.6V-Flash | `eval_Glm46v.py` |
| 2 | InternVL2_5-8B | `eval_internvl25.py` |
| 3 | deepseek-vl2-small | `eval_deepseek_vl2.py` |
| 4 | Qwen2.5-VL-7B-Instruct | `eval_qwen25.py` |
| 5 | CogVLM2-19B | `eval_Cogvlm.py` |
| 6–8 | Qwen2-VL-14B / Pixtral-12B / Llama-3.2-11B | reserved — evaluator not yet implemented |

> **Note:** IDs 6, 7, and 8 (Qwen2-VL-14B-Instruct, Pixtral-12B-2409, Llama-3.2-11B-Vision-Instruct) are reserved slots. Their evaluator files do not yet exist; refer to the "Adding a New VLM" section below to implement them.

### Evaluation Modes

| Mode | Models | Description |
|------|--------|-------------|
| **Pipeline** | Qwen · GLM · CogVLM | Background thread loads images in parallel; GPU inference overlaps with I/O, hiding per-sample I/O latency |
| **Sequential** | InternVL2.5 · DeepSeek-VL2 | Model bundles I/O and GPU pre-processing internally; samples are processed one at a time |

Results are written to `DATA/question_results/<model_name>/Evaluator_<question_type>_<language>.xlsx`.

---

## Benchmark Results

Accuracy (%) across all question types. Random-guess baseline is 20% (5-choice). Full result files are in `DATA/question_results/`.

### Chinese prompts (`cn`)

| Model | q1 | q2 | q3 | q4 | q5 | q6 | q7 |
|-------|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B-Instruct | **95.1** | **93.9** | **81.9** | **64.1** | **80.7** | 74.2 | **80.7** |
| Qwen2.5-VL-7B-Instruct | 95.6 | 92.2 | 76.7 | 63.7 | 78.6 | **75.3** | 81.5 |
| GLM-4.6V-Flash | 93.2 | 89.3 | 75.9 | 64.7 | 65.9 | 71.8 | 73.1 |
| InternVL2_5-8B | 90.7 | 82.7 | 75.1 | 56.0 | 73.7 | 70.4 | 71.2 |
| deepseek-vl2-small | 85.6 | 79.2 | 68.6 | 61.6 | 66.9 | 67.3 | 68.1 |
| CogVLM2-19B | 81.0 | — | 50.4 | 47.0 | 52.3 | 54.8 | 52.3 |

### English prompts (`en`)

| Model | q1 | q2 | q3 | q4 | q5 | q6 | q7 |
|-------|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B-Instruct | **95.0** | **92.0** | **79.6** | **64.7** | 76.0 | **74.1** | **76.6** |
| Qwen2.5-VL-7B-Instruct | 94.6 | 89.5 | 75.0 | 62.5 | **78.0** | 72.8 | 75.8 |
| InternVL2_5-8B | 87.5 | 78.9 | 69.4 | 60.7 | 71.1 | 64.1 | 63.9 |
| deepseek-vl2-small | 84.9 | 70.3 | 62.3 | 60.8 | 62.0 | 59.4 | 62.3 |
| GLM-4.6V-Flash | 68.4 | 75.7 | 64.5 | 44.0 | 70.3 | 57.6 | 63.9 |
| CogVLM2-19B | 77.9 | — | 55.9 | 51.8 | 48.8 | 53.2 | 49.8 |

> "—" indicates the combination was not evaluated.

---

## Adding a New VLM

1. Create `Vlm_Eval/eval_<model>.py`, following an existing implementation (e.g. [eval_qwen25.py](Vlm_Eval/eval_qwen25.py)).

   **Pipeline mode** — implement:
   - `choice_single_img_promote(promote, question, options, pil_img, language_type)` → `inputs`
   - `choice_multi_img_promote(promote, question, pil_imgs, language_type)` → `(inputs, letters)`
   - `generate_text(inputs, max_new_tokens)` → `str`

   **Sequential mode** — implement:
   - `eval_choice_single_img(promote, img_path, question, options, max_new_tokens, language_type)` → `(raw_str, letter_str)`
   - `eval_choice_multi_img(promote, img_paths, question, max_new_tokens, language_type)` → `(raw_str, letter_str)`

2. Add the model name to `model_id_list` in [VLM_test_parallel.py](VLM_test_parallel.py).

3. Add the corresponding `elif` branch in `build_evaluator()` to instantiate the new evaluator.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | GPU inference |
| `transformers` ≥ 4.44 | Model loading (AutoModel / AutoProcessor) |
| `accelerate` | Device mapping / multi-GPU support |
| `Pillow` | Image loading |
| `tqdm` | Progress bars |
| `pandas` / `openpyxl` | Result saving and reading (`.xlsx`) |
| `qwen-vl-utils` | Qwen-VL vision input preprocessing *(optional)* |
| `google-cloud-translate` | Translation helpers in `data_preprocess.py` *(optional)* |

---

## License

The benchmark code is released under the **MIT License**.  
Images are sourced from Weibo; copyright belongs to the original rights holders.
