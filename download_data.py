# -*- encoding: utf-8 -*-
# @Time    :   2024/11/23 12:35:12
# @File    :   download_data.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   下载数据


import os
import traceback
from tqdm import tqdm
from PIL import Image
from typing import List
from datasets import load_dataset

def downloadData(dataset_name, split_name, data_size=5000):
    dataset = load_dataset(dataset_name)
    try:
        train_dataset = dataset[split_name]
    except:
        print(f"{dataset_name}没有{split_name}的切割数据")
        return 
    names = train_dataset.features["label"].names \
                            if hasattr(train_dataset.features["label"], "names") \
                            else [] # 这里可能某些数据集不提供names 请手动修改
    train_dataset = train_dataset.shuffle(seed=42)
    train_dataset = train_dataset[:data_size]
    img_dir = f"data/{split_name}"

    os.makedirs(img_dir, exist_ok=True)
    img_list:List[Image.Image] = train_dataset["image"]
    label_list = train_dataset["label"]
    save_path = "configs"
    os.makedirs(save_path, exist_ok=True)
    f = open(os.path.join(save_path, f"{split_name}.txt"), "w", encoding="utf-8")
    try:
        for index, item in tqdm(enumerate(zip(img_list, label_list)), total=max(len(img_list), data_size)):
            img, label = item
            if names:
                label = names[label]
            img_save_path = os.path.join(img_dir, f"{index}.jpg")
            img = img.convert("RGB")
            img.save(img_save_path)
            f.write(f"{img_save_path}\t{label}\n")
        f.close()
    except:
        print(traceback.format_exc())
        

def downloadLabelConfig(dataset_name, split_name="train"):
    dataset = load_dataset(dataset_name, split=split_name)
    names = dataset.features["label"].names \
                        if hasattr(dataset.features["label"], "names") \
                        else [] # 这里可能某些数据集不提供names 请手动修改
    with open("configs/label_config.txt", "w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")
        

if __name__ == "__main__":
    dataset_name = "food101"
    downloadData(dataset_name, "train")
    downloadData(dataset_name, "validation")
    downloadData(dataset_name, "test")
    downloadLabelConfig(dataset_name)