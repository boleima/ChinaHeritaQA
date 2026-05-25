"""Generate question q6: Which main-function description correctly matches this image?"""
import os
import json
import random
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from utils.data_preprocess import map_heritage_infos_by_keys,get_province_city,random_exclude,openJson,generate_option,data_filter

if __name__ == "__main__":
    '''
    Generate single-choice questions.
    Question: Which description of the main function is correct for this image?
    Options list: [
        A: correct answer,
        B: same-type heritage site,
        C: same-province heritage site,
        D: any Chinese heritage site,
        E: non-Chinese heritage site
    ]
    '''
    heritage_meta = openJson(config.HERITAGE_META)
    pro_heritage_file = openJson(config.HERITAGE_CITY)   # sorted by province
    type_heritage_file = openJson(config.HERITAGE_TYPE)  # sorted by type
    world_heritage_info = openJson(config.WORLD_HERITAGE_INFO)
    save_name = config.QUESTION_FILES["q6"]
    rt_img_weibo = config.IMG_PREFIX_CHINA

    heritage_meta = data_filter(heritage_meta, ["Number of Images"], 0, False)
    heritage_meta = data_filter(heritage_meta, ["Main_function",'cn'], 'none', False)
    world_heritage_info = data_filter(world_heritage_info, ["Main_function",'cn'], 'none', False)

    all_keys = list(heritage_meta.keys())
    question_one_info = {}
    question_options_info = []
    idx = 0

    for key, val in tqdm(heritage_meta.items()):
        Main_function = val['Main_function']
        img_url_list = val['Image URLs']
        option_A = generate_option(cn_text=Main_function['cn'], en_text=Main_function['en'], is_true=True)
        for img_url in img_url_list:
            save_type = {}

            # option_C: same-province distractor
            city_str = val['City']['cn']
            tmp_keys_list = all_keys
            if city_str != "":
                pro, city = get_province_city(city_str)
                candidate_lit = list(set(heritage_meta.keys()) & set(pro_heritage_file[pro]))
                if len(candidate_lit) > 1:
                    tmp_keys_list = list(filter(lambda k: k != key, candidate_lit))
            tmp_keys_infos = map_heritage_infos_by_keys(heritage_meta, tmp_keys_list)
            option_C_text = random_exclude(tmp_keys_infos, [Main_function['cn']], ['Main_function','cn'])
            option_C = generate_option(cn_text=option_C_text['cn'], en_text=option_C_text['en'])

            # option_B: same-type distractor (falls back to full pool if insufficient)
            type_str = val['Type']['cn']
            type_keys = list(set(heritage_meta.keys()) & set(type_heritage_file[type_str]))
            if len([k for k in type_keys if heritage_meta[k]['Main_function']['cn'] not in (Main_function['cn'], option_C_text['cn'])]) == 0:
                type_keys = all_keys
            tmp_keys_infos = map_heritage_infos_by_keys(heritage_meta, type_keys)
            option_B_text = random_exclude(tmp_keys_infos, [Main_function['cn'], option_C_text['cn']], ['Main_function','cn'])
            option_B = generate_option(cn_text=option_B_text['cn'], en_text=option_B_text['en'])

            # option_D: any Chinese heritage site
            option_D_text = random_exclude(heritage_meta, [Main_function['cn'], option_C_text['cn'], option_B_text['cn']], ['Main_function','cn'])
            option_D = generate_option(cn_text=option_D_text['cn'], en_text=option_D_text['en'])

            # option_E: non-Chinese site (same type preferred; falls back to full pool)
            type_worlds = data_filter(world_heritage_info, ['Type','cn'], type_str, False)
            if not type_worlds:
                type_worlds = world_heritage_info
            option_E_text = random_exclude(type_worlds, [Main_function['cn'], option_C_text['cn'], option_B_text['cn'], option_D_text['cn']], ['Main_function','cn'])
            option_E = generate_option(cn_text=option_E_text['cn'], en_text=option_E_text['en'])

            options_list = [option_A, option_B, option_C, option_D, option_E]
            # combine options
            save_type['index']=idx
            idx+=1
            save_type['img_url'] = os.path.join(rt_img_weibo,img_url)
            random.shuffle(options_list)
            save_type['options'] = options_list
            question_options_info.append(save_type)
    question_one_info['promote'] = {
        'cn': '你是一个专业的文化与自然遗产识别系统。请仔细观察题目中的图片，并认真阅读选项 A、B、C、D、E 的内容，结合图片信息从中选出唯一正确答案。\n要求：\n- 只输出正确选项的字母（A / B / C / D / E）\n- 不要添加任何解释、标点或其他内容\n示例回答：A\n',
        'en': 'You are a professional cultural and natural heritage recognition system. Please carefully examine the image in the question and read through options A, B, C, D, and E, then identify the single correct answer based on the visual content.\nRequirements:\n- Output only the letter of the correct option (A / B / C / D / E)\n- Do not include any explanation, punctuation, or additional content\nExample answer: A\n'
    }
    question_one_info['question_info']={
        "question_type": {'cn':"单选题" , 'en':"Multiple-choice question"},
        "question_content":{'cn':'关于该图片主要的功能介绍正确的是?','en':'Which description of the main function of this picture is correct?'},
        "is_img_in_question":True,
        "options_INFO": question_options_info
    }

    with open(save_name, "w", encoding="utf-8") as f:
        json.dump(question_one_info, f, ensure_ascii=False, indent=4)
