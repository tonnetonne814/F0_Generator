import yaml
import sys
import os
import pickle
import requests

def read_yml(yml_filepath):
    with open(yml_filepath, 'r', encoding="utf-8") as yml:
        config = yaml.safe_load(yml)
    return config

def write_yaml(yml_filepath, save_dict):
    with open(yml_filepath,'w', encoding="utf-8")as f:
        yaml.dump(save_dict, f, default_flow_style=False, allow_unicode=True)
    return 0

def read_txt(txt_path):
    with open(txt_path , mode="r", encoding="utf-8") as f:
        txt = f.read()
    return txt

def write_txt (txt_path, object):
    # 文字の時
    if isinstance(object, str) is True:
        with open(txt_path, mode="w", encoding="utf-8") as f:
            f.write(object)
    # リストの時
    elif isinstance(object, list) is True:
        with open(txt_path, mode="w", encoding="utf-8") as f:
            f.writelines(object)

# ライブラリを読み込む
def read_libraries(folder_path):
    folder_list = os.listdir(folder_path)
    for name in folder_list:
        lib_path = os.path.join(folder_path, name)
        sys.path.append(lib_path)

def download_file(url, dst_path):
    response = requests.get(url)
    image = response.content

    with open(dst_path, "wb") as aaa:
        aaa.write(image)

# pickle化してファイルに書き込み
def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

# 非pickle化
def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

# exeにぶち込む用
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)