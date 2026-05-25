import random
import json
import os
def translate_text(text, target_language='en'):
    from google.cloud import translate_v2 as translate
    """
    Translate text using the Google Cloud Translation API.
    :param text: text to translate
    :param target_language: target language code (e.g. 'en', 'zh-CN', 'de')
    :return: translated text string
    """
    # create translation client
    translate_client = translate.Client()
    # ensure input is a string
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    # perform translation
    result = translate_client.translate(text, target_language=target_language)

    return result['translatedText']
def random_exclude(data, ext=[],subkeys_use=None):
    '''Returns a random choice from the list excluding the specified item.'''
    candidates=[]
    if subkeys_use is not None:
        for k,v in data.items():
            tmp_f=v
            tmp=None
            for i in range(len(subkeys_use)):
                if tmp:
                    tmp_f=tmp
                tmp=tmp_f[subkeys_use[i]]
            if tmp in ext:
                continue
            candidates.append(tmp_f)
    else:
        candidates = list(set(data) - set(ext))
    return random.choice(candidates)




def get_province_city(city_str):
    if city_str in ["北京市", "天津市", "上海市", "重庆市","澳门特别行政区", "香港特别行政区"]:
        pro = city_str
        city = city_str
    elif "省" in city_str:
        city_str = city_str.split("省")
        pro = city_str[0]+"省"
        city = city_str[1]
    elif "自治区" in city_str:
        city_str = city_str.split("自治区")
        pro = city_str[0]+"自治区"
        city = city_str[1]
    else:
        print(city_str)
    return pro, city

def city_type_filter(xx_meta,meta_data,category,throlder):
    result={}
    for k,v in xx_meta.items():
        c_list = []
        for kk in v:
            if meta_data[kk][category]==throlder:
                continue
            c_list.append(kk)
        result[k]=c_list
    return result


def generate_option(cn_text=None,en_text=None,img_url=None,is_true=False):
    '''Generates a dictionary with the text and an optional issue flag.'''
    option={}
    option['text']={
        "cn":cn_text,
        "en":en_text
    }
    option["img_url"] = img_url
    option["is_true"] = is_true
    return option

def openJson(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta
def saveJson(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def map_heritage_info(heritage_meta,option):
    return heritage_meta[option]
def map_heritage_infos_by_keys(heritage_meta,keys):
    result={}
    for k,v in heritage_meta.items():
        if k in keys:
            result[k]=v
    return result

def map_heritage_name(heritage_meta,option,language='both'):
    if language=='both':
        return heritage_meta[option]['Name']
    else:
        return heritage_meta[option]['Name'][language]
def map_heritage_des(heritage_meta,option,language='both'):
    if language=='both':
        return heritage_meta[option]['Description']
    else:
        return heritage_meta[option]['Description'][language]
def map_category_heritage(heritage_meta,option,category,language='both'):
    if language=='both':
        return heritage_meta[option][category]
    else:
        return heritage_meta[option][category][language]

def data_filter(heritage_meta,categories,throlder,only_k=True):
    result_k = []
    result_all={}
    for k,v in heritage_meta.items():
        if len(categories)==2:
            if v[categories[0]][categories[1]]==throlder:
                    continue
        else:
            if v[categories[0]]==throlder:
                    continue
        if only_k:
            result_k.append(k)
        else:
            result_all[k]=v
    return result_k if only_k else result_all

def data_choice(heritage_meta,categories,throlder,only_k=True):
    result_k = []
    result_all={}
    for k,v in heritage_meta.items():
        if len(categories)==2:
            if v[categories[0]][categories[1]]!=throlder:
                    continue
        else:
            if v[categories[0]]!=throlder:
                    continue
        if only_k:
            result_k.append(k)
        else:
            result_all[k]=v
    return result_k if only_k else result_all


def update_image_urls(json_path, image_data_dir, save=True):
    """
    Scan Image_data/<site_key>/ folders and update "Image URLs" and
    "Number of Images" in the heritage meta JSON to reflect files
    that actually exist on disk.
    """
    meta = openJson(json_path)
    for site_key, site_info in meta.items():
        site_dir = os.path.join(image_data_dir, site_key)
        if not os.path.isdir(site_dir):
            print(f"[SKIP] directory not found: {site_dir}")
            continue
        files = sorted(
            f for f in os.listdir(site_dir)
            if os.path.isfile(os.path.join(site_dir, f))
        )
        site_info["Image URLs"] = [f"{site_key}\\{f}" for f in files]
        site_info["Number of Images"] = len(files)
    if save:
        saveJson(meta, json_path)
        print(f"Saved updated JSON to {json_path}")
    return meta
