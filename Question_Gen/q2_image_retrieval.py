"""Generate question q2: Which of the following images was likely taken at the given heritage site?"""
import os
import json
import random
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from utils.data_preprocess import get_province_city,random_exclude,openJson,generate_option,map_heritage_info,data_filter,city_type_filter


if __name__ == "__main__":
    '''
    Generate single-choice questions.
    Question: Which of the following images was likely taken at the {site name}?
    Options list: [
        A: correct answer (contains the correct image),
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

    save_name = config.QUESTION_FILES["q2"]
    rt_img_weibo = config.IMG_PREFIX_CHINA
    rt_img_worlds = config.IMG_PREFIX_WORLD

    pro_heritage_file = city_type_filter(pro_heritage_file, heritage_meta, "Number of Images", 0)
    type_heritage_file = city_type_filter(type_heritage_file, heritage_meta, "Number of Images", 0)
    heritage_meta = data_filter(heritage_meta, ["Number of Images"], 0, False)

    all_keys = list(heritage_meta.keys())
    idx=0
    question_one_info = {}
    question_options_info = []

    for key,val in tqdm(heritage_meta.items()):
        Name = val['Name']
        for img in val["Image URLs"]:
            option_A = generate_option(cn_text=Name['cn'],en_text=Name['en'],img_url=os.path.join(rt_img_weibo,img), is_true=True)
            save_type = {}

            # option_C: same-province distractor
            city_str = val['City']['cn']
            tmp_keys_list = all_keys
            if city_str !="":
                    pro,city = get_province_city(city_str)
                    candidate_lit = pro_heritage_file[pro]
                    if len(candidate_lit) > 1:
                        tmp_keys_list = list(filter(lambda k: k != key, candidate_lit))
            option_C_key = random_exclude(tmp_keys_list,[key])
            option_C_info = map_heritage_info(heritage_meta,option_C_key)
            option_C = generate_option(cn_text=option_C_info['Name']['cn'],en_text=option_C_info['Name']['en'],img_url=os.path.join(rt_img_weibo,random_exclude(option_C_info["Image URLs"])),is_true=False)

            # option_B: same-type distractor
            type_str = val['Type']['cn']
            option_B_key = random_exclude(type_heritage_file[type_str], [key, option_C_key])
            option_B_info = map_heritage_info(heritage_meta,option_B_key)
            option_B = generate_option(cn_text=option_B_info['Name']['cn'],en_text=option_B_info['Name']['en'],img_url=os.path.join(rt_img_weibo,random_exclude(option_B_info["Image URLs"])),is_true=False)

            # option_D: any Chinese heritage site
            option_D_key = random_exclude(all_keys, [key, option_B_key, option_C_key])
            option_D_info = map_heritage_info(heritage_meta,option_D_key)
            option_D = generate_option(cn_text=option_D_info['Name']['cn'],en_text=option_D_info['Name']['en'],img_url=os.path.join(rt_img_weibo,random_exclude(option_D_info["Image URLs"])),is_true=False)

            # option_E: non-Chinese heritage site
            option_E_key = random_exclude(world_heritage_info.keys(), [])
            option_E_info= map_heritage_info(world_heritage_info,option_E_key)
            option_E = generate_option(cn_text=option_E_info['Name']['cn'],en_text=option_E_info['Name']['en'],img_url=os.path.join(rt_img_worlds,option_E_key,random_exclude(option_E_info["Image URLs"])),is_true=False)

            options_list = [option_A, option_B, option_C, option_D,option_E]

            # combine options
            save_type['index']=idx
            idx+=1
            save_type['Name'] = Name
            random.shuffle(options_list)
            save_type['options'] = options_list
            question_options_info.append(save_type)

    question_one_info['promote'] = {
        'cn': '你是一个专业的文化与自然遗产识别系统。请仔细观察题目中的图片，并认真阅读选项 A、B、C、D、E 的内容，结合图片信息从中选出唯一正确答案。\n要求：\n- 只输出正确选项的字母（A / B / C / D / E）\n- 不要添加任何解释、标点或其他内容\n示例回答：A\n',
        'en': 'You are a professional cultural and natural heritage recognition system. Please carefully examine the image in the question and read through options A, B, C, D, and E, then identify the single correct answer based on the visual content.\nRequirements:\n- Output only the letter of the correct option (A / B / C / D / E)\n- Do not include any explanation, punctuation, or additional content\nExample answer: A\n'
    }
    question_one_info['question_info']={
        "question_type": {'cn':"单选题" , 'en':"Multiple-choice question"},
        "question_content":{'cn':'以下哪个图片可能是在{name}拍摄的?','en':'Which of the following images was likely taken at the {name}?'},
        "is_img_in_question":False,
        "options_INFO": question_options_info
    }

    with open(save_name, "w", encoding="utf-8") as f:
        json.dump(question_one_info, f, ensure_ascii=False, indent=4)
