"""
config.py — ChinaHeritaQA global path configuration

All scripts read paths from this file; no hardcoding elsewhere.
To change a path, edit only this file.
"""
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# Project root directory (resolved automatically from this file's location)
# ══════════════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parent

# ══════════════════════════════════════════════════════════════════════════════
# HPC / remote server paths  ← must be updated before deployment
# ══════════════════════════════════════════════════════════════════════════════

# Root directory that contains all downloaded model weight folders.
# Each subfolder name must match an entry in model_id_list of VLM_test_parallel.py.
# Example (Linux HPC): Path("/data/models")
# Example (Windows):   Path(r"D:\models")
MODEL_ROOT    = Path("/path/to/model/weights/root")

# Root directory on a large-capacity disk used for triton / HuggingFace caches.
# Must point to a partition with sufficient free space (≥ 50 GB recommended).
# Example (Linux HPC): Path("/scratch/username")
# Example (Windows):   Path(r"E:\cache")
BIG_DISK_ROOT = Path("/path/to/large/disk/scratch")

# HPC cache directories (used by eval_Cogvlm.py)
TRITON_CACHE  = BIG_DISK_ROOT / "triton_cache"
HF_CACHE      = BIG_DISK_ROOT / "hf_cache"

# ══════════════════════════════════════════════════════════════════════════════
# DATA directory
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = ROOT / "DATA"

# ── Metadata JSON files ───────────────────────────────────────────────────────
HERITAGE_META        = DATA_DIR / "heritage_meta_weibo_V1.json"
HERITAGE_CITY        = DATA_DIR / "heritage_city.json"
HERITAGE_TYPE        = DATA_DIR / "heritage_type.json"
WORLD_HERITAGE_INFO  = DATA_DIR / "world_heritage_info_V1.json"
DYNAST_LIST          = DATA_DIR / "dynast_list_V1.json"
HERITAGE_BRIEF_INTRO = DATA_DIR / "heritage_brief_intro.json"

# ── Image directories ─────────────────────────────────────────────────────────
IMAGE_ROOT      = DATA_DIR / "Images"              # image root used during evaluation
IMAGE_DATA_DIR  = IMAGE_ROOT / "Image_data"        # Chinese heritage site images
WORLDS_DATA_DIR = IMAGE_ROOT / "worlds_data"       # world heritage site images

# ── Image URL prefixes stored in question JSON files (relative to IMAGE_ROOT) ─
IMG_PREFIX_CHINA = "Image_data"   # prefix for Chinese heritage site images
IMG_PREFIX_WORLD = "worlds_data"  # prefix for world heritage site images

# ── Question JSON files ───────────────────────────────────────────────────────
QUESTION_DIR   = DATA_DIR / "quesion_info"
QUESTION_FILES = {
    q: QUESTION_DIR / f"{q}.json"
    for q in ("q1", "q2", "q3", "q4", "q5", "q6", "q7")
}

# ── Evaluation results directory ──────────────────────────────────────────────
RESULTS_DIR = DATA_DIR / "question_results"
