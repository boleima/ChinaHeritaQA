"""Generate question q3: In which dynasty was the architectural complex in this image likely built?"""
import os
import json
import random
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from utils.data_preprocess import random_exclude,openJson,generate_option,data_choice,data_filter

def normalize(text):
    return text.split('(')[0]

if __name__ == "__main__":
    '''
    Generate single-choice questions.
    Question: In which dynasty was the architectural complex in this image likely built?
    Options list: [
        A: correct answer,
        B: random Chinese dynasty,
        C: dynasty of same-type architecture,
        D: random Chinese dynasty,
        E: European calendar era
    ]
    '''
    heritage_meta = openJson(config.HERITAGE_META)
    dynast_list = openJson(config.DYNAST_LIST)
    save_name = config.QUESTION_FILES["q4"]
    rt_img_weibo = config.IMG_PREFIX_CHINA

    heritage_meta = data_filter(heritage_meta, ["Number of Images"], 0, False)
    heritage_meta = data_filter(heritage_meta, ['Dynast','cn'], "无具体朝代", False)

    all_keys = list(heritage_meta.keys())
    question_one_info = {}
    question_options_info = []
    idx = 0

    for key, val in tqdm(heritage_meta.items()):
        dynast = val['Dynast']
        img_url_list = val['Image URLs']
        option_A = generate_option(cn_text=normalize(dynast['cn']), en_text=normalize(dynast['en']), is_true=True)
        for img_url in img_url_list:
            save_type = {}
            # option_C: dynasty of same-type architecture (falls back to dynasty list if pool is insufficient)
            type_str = val['Type']['cn']
            type_heritage_pool = data_choice(heritage_meta, ['Type','cn'], type_str, False)
            try:
                option_C_text = random_exclude(type_heritage_pool, [dynast['cn']], ['Dynast','cn'])
            except IndexError:
                option_C_text = random_exclude(dynast_list['chinese_dynasties'], [dynast['cn']], ['cn'])
            option_C = generate_option(cn_text=normalize(option_C_text['cn']),en_text=normalize(option_C_text['en']))
            # option_B: random Chinese dynasty
            option_B_text = random_exclude(dynast_list['chinese_dynasties'],[dynast['cn'],option_C_text['cn']],['cn'])
            option_B = generate_option(cn_text=normalize(option_B_text['cn']),en_text=normalize(option_B_text['en']))

            # option_D: random Chinese dynasty
            option_D_text = random_exclude(dynast_list['chinese_dynasties'],[dynast['cn'], option_C_text['cn'],option_B_text['cn']],['cn'])
            option_D = generate_option(cn_text=normalize(option_D_text['cn']),en_text=normalize(option_D_text['en']))

            # option_E: European calendar era
            option_E_text = random_exclude(dynast_list['Euro_dynasties'],[], ['cn'])
            option_E = generate_option(cn_text=normalize(option_E_text['cn']),en_text=normalize(option_E_text['en']))

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
        "question_content":{'cn':'该图片中的建筑群可能建于哪个朝代?','en':'In which dynasty might the building complex in this picture have been built?'},
        "is_img_in_question":True,
        "options_INFO": question_options_info
    }
    with open(save_name, "w", encoding="utf-8") as f:
        json.dump(question_one_info, f, ensure_ascii=False, indent=4)
