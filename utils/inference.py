import os
import re
import csv
import json
import pandas as pd
from PIL import Image
LETTER_LIST = [chr(ord("A") + i) for i in range(26)]
LETTER_RE = re.compile(r"\b([A-Z])\b")  # extract A/B/C/... option letters


def letters_for_n(n):
    return LETTER_LIST[:n]
def get_img(img_url,image_root):
    img_rel = str(img_url).replace("\\\\", "/").replace("\\", "/")
    image_path = os.path.join(image_root, img_rel) if image_root else img_rel
    return image_path
def dataset_load(data_path,image_root,language='cn'):
    with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)   
    q_promote = data['promote'][language]
    qinfo = data["question_info"]
    q_content = qinfo['question_content'][language]
    entries = qinfo.get("options_INFO", [])

    is_img_in_question = qinfo.get('is_img_in_question', True)

    for entry in entries:
        opts = entry.get("options", [])
        options = [o['text'][language] for o in opts]
        true_idx = [i for i, o in enumerate(opts) if o.get("is_true", False)]
        index = entry.get('index',-1)
        if is_img_in_question:
            # Case 1: single-image question (image in the question, options are text)
            image_path = get_img(entry.get("img_url", ""), image_root)
            gold_letters = [letters_for_n(len(options))[i] for i in true_idx]
            yield index,q_promote,image_path, q_content, options, gold_letters
        else:
            # Case 2: multi-image choice question (each option is an image)
            image_list = [get_img(o.get("img_url", ""), image_root) for o in opts]
            q_text = q_content.replace('{name}', entry["Name"][language])
            gold_letters = [letters_for_n(len(image_list))[i] for i in true_idx]
            yield index,q_promote,image_list, q_text, options, gold_letters
            

def is_correct(pred_letters,gold_letters):
    # Accuracy: for single-choice, only the first letter is checked
    if pred_letters is None:  # safety check
        return False
    pred_letters = pred_letters[:1]
    ok = (len(pred_letters) == 1 and len(gold_letters) == 1 and pred_letters[0] == gold_letters[0])
    return ok

def save_as_(data,output_file):
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    if output_file.lower().endswith(".xlsx"):
        pd.DataFrame(data).to_excel(output_file, index=False)
    else:
        with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
