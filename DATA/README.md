# DATA

This folder contains all metadata and example data for the ChinaHeritaQA benchmark.

## Contents

| File / Folder | Description |
|---|---|
| `heritage_meta_weibo_V1.json` | Metadata for Chinese UNESCO World Heritage Sites |
| `world_heritage_info_V1.json` | Metadata for non-Chinese heritage sites (distractors) |
| `heritage_city.json` | Heritage sites indexed by province |
| `heritage_type.json` | Heritage sites indexed by type |
| `dynast_list_V1.json` | Chinese dynasties and European historical eras |
| `heritage_brief_intro.json` | Brief descriptions used for question type q3 |
| `quesion_info/` | Generated question JSONs (`q1.json` … `q7.json`) |
| `question_results/` | Evaluation outputs (one `.xlsx` per model × question type × language) |
| `Images/` | **Sample images only** — see note below |

## Note on Image Data

A small set of example images is provided in `Images/` for reference and to allow the repository structure to be inspected.

Due to the large volume of visual data (**over 7 GB**), the full image dataset will be released on an external repository upon paper publication.

### Image folder structure (full dataset)

```
Images/
├── Image_data/          # Chinese heritage site images (sourced from Weibo)
│   └── <site_name>/
│       ├── <post_id>_1.jpg
│       └── ...
└── worlds_data/         # Non-Chinese World Heritage Site images (distractors)
    └── <site_name>/
        ├── 1.jpg
        └── ...
```
