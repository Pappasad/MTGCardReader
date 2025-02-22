import math
import os
import re
import time
import json
from PIL import Image
import numpy as np
import torch
import torchvision
from code.APutil import ALL_DIR, imread, imshow
# from JR_single_easyocr import SimpleOCR
from skimage.morphology import opening
from code.APPaddleOcr import NewOCR

img_folder = "test_img_storage/" # "katie_folder/"    # save debug images in this folder
CROP_DIR = os.path.join('data', 'images', 'crop') # save cropped images in this folder
debug = 1

def convert_json_file(file: str) -> dict:
    json_file = open(file, 'r')
    json_data = json.load(json_file)
    if (type(json_data) == dict):
        return json_data
    # print(len(json_data))
    diction = {}
    for item in json_data:
        if item.get("lang") != "en":
            continue
        diction[item.get('Actual_Title').lower()] = item
    return diction

def find_accuracy(out: dict) -> None:
    # Find current accuracy
    accuracy = 0
    file = open(f"{img_folder}Wrong_Filepaths.text", "a")
    for entry in out.keys():
        item = out.get(entry)
        if (item.get('Actual_Title') == None or item.get('Predicted_Title') == None):
            # print(item)
            continue
        cleaned_predict = re.sub(r'[^\w\s]', '', item.get('Predicted_Title'))
        cleaned_predict = re.sub(r"\s+", " ", cleaned_predict).strip()
        cleaned_actual = re.sub(r'[^\w\s]', '', item.get('Actual_Title'))
        cleaned_actual = re.sub(r"\s+", " ", cleaned_actual).strip()
        if (item.get('Actual_Title').lower() == item.get('Predicted_Title').lower()
            or cleaned_actual.lower() == cleaned_predict.lower()):
            accuracy += 1
        elif (debug):
            file.write(item.get('File') + "\n")
            # print(item)
    accuracy /= len(out.keys())
    print(accuracy)

def convert_list_to_dict() -> dict:
    # This code converts from a list of strings to a json object
    # Searchable by actual name of the card. 
    out = {}
    with open(f"{img_folder}out.json", 'r') as file:
        data = [json.loads(line.strip()) for line in file]

    for item in data:
        out[item.get('Actual_Title')] = item

    print(len(out.keys()))
    out_file = open(f"{img_folder}clean.json", "w")
    json.dump(out, out_file)
    return out

def get_some_files() -> None:
    num_imgs = 50
    for i in range(0, num_imgs):
        file = os.path.join(ALL_DIR, os.listdir(ALL_DIR)[i])
        image = Image.open(file)
        img_np = np.array(image)
        imshow(img_np, f"{img_folder}{os.listdir(ALL_DIR)[i]}")

def text_handler(text: str):
    text = re.sub(r'[^\w\s]', '', text) # remove special characters
    text = re.sub(r'\d+', '', text)     # remove numbers
    text = text.replace('\n', '')       # remove newlines
    return text.strip()                 # remove leading and tailing spaces

def getSaveFile(specific_file: str) -> None:
    file = os.path.join(ALL_DIR, specific_file)
    image = Image.open(file)
    img_np = np.array(image)
    imshow(img_np, f"{img_folder}{specific_file}")

def np_crop(img: np.array, top: int, left: int, height: int, width: int):
    return img[top : top+height, left : left+width]

if __name__ == "__main__":
    start_time = time.time()

    # out = convert_list_to_dict()

    # get_some_files()

    file_data = open(f"{img_folder}out.json", 'r')
    out = json.load(file_data)
    print(len(out.keys()))
    find_accuracy(out)

    # specific_file = "Blank_Card_-_ptc_0.jpg"
    # OCR = SimpleOCR()
    # offset = 0
    # file = open(f"{img_folder}Wrong_Filepaths.text", 'r')
    # for index, path in enumerate(file):
    #     if offset <= index <= (offset + 5):
    #         # print(path.replace('\n', ''))
    #         path = path.replace('\n', '')
    #         getSaveFile(path)

    #         file = os.path.join(ALL_DIR, path)
    #         data = imread(file)
    #         text = OCR(data)
            # imshow(img_np, f"{img_folder}Crop{path}")
            # print(text)

    # file = "./data/images/all/Blank_Card_-_ptc_0.jpg"
    # OCR = NewOCR()
    # prin = OCR(file)
    # print(type(prin))
    # print("`" + prin + "`")


    print(f"Execution time: {time.time() - start_time} seconds")
    