import os, re, shutil, time, sys, json
from torchvision.io import read_image as imread
from code.APutil import ALL_DIR, imshow
# OCR
# import easyocr
# import numpy as np
import torch, string
import collections as ct
from code.APEasyOcr import SimpleOCR
# Postprocessing
from fuzzywuzzy import fuzz


debug = 1                           # do you want nothing saved, the last step, or everything [0, 1, 2]
num_imgs = int(sys.argv[1])              # len(os.listdir(ALL_DIR))/10 
                                    # how many images to test (can use all in dir, but DAYS) # 3000 is ok
img_folder = 'test_img_storage/'    # save debug images in this folder
CROP_DIR = os.path.join('data', 'images', 'crop') # save cropped images in this folder
special_chars = string.punctuation
out_file = f'{img_folder}out.json'

def find_best_match(json_data, target_string):
    best_match = None
    best_score = 0
    for key in json_data.keys():
        score = fuzz.ratio(key, target_string)
        if score > best_score:
            best_match = key
            best_score = score
    return best_match, best_score

def convert_json_file(file: str):
    json_file = open(file, 'r')
    json_data = json.load(json_file)
    # print(len(json_data))
    diction = {}
    for item in json_data:
        if item.get('lang') != 'en':
            continue
        diction[item.get('name').lower()] = item
    return diction

def moreThanTitle(text: list[str], json_data: dict):
    text = ' '.join(text[:3])
    best_match, confidence = find_best_match(json_data, text.lower())
    confidence /= 100.0
    return best_match, confidence, text

def rotateImage(data: torch.Tensor, json_data: dict):
    # print(os.listdir(ALL_DIR)[i] + ' ' + text + ' ' + best_match + ' ' + str(confidence))
    data = torch.rot90(data, k=-1, dims=[1, 2])
    text = OCR(data)
    if (text != []):
        best_match, confidence = find_best_match(json_data, text[0].lower())
        confidence /= 100.0
        if (confidence < 0.6):
            best_match_2, confidence_2, text_2 = moreThanTitle(text, json_data)
            if (confidence_2 > confidence):
                best_match = best_match_2
                confidence = confidence_2
                text = text_2
            else:
                text = text[0]
    else:
        return "", 0.0, ""
    return best_match, confidence, text

def recognize(i: int, json_data: dict, OCR: SimpleOCR):
        # Get actual card title
        actual_title = os.listdir(ALL_DIR)[i]
        actual_title = actual_title[:actual_title.rfind('_-_')].replace('_', ' ').strip()

        # Get image
        file = os.path.join(ALL_DIR, os.listdir(ALL_DIR)[i])
        data = imread(file)
        
        # OCR
        org_text = OCR(data)
        # print(text)
        if (org_text != []):
            text = org_text[0]
        else:
            text = ''
        best_match, confidence = find_best_match(json_data, text.lower())
        confidence /= 100.0
        # if (confidence < 0.5):
        #     for item in org_text:
        #         best_match_list, confidence_list = find_best_match(json_data, item.lower())
        #         if confidence_list/100 > confidence:
        #             confidence = confidence_list/100
        #             best_match = best_match_list
        #             text = item
        num_spec = sum(v for k, v in ct.Counter(text).items() if k in special_chars)
        if (len(text) < 4 or num_spec > 3 or confidence < 0.6):
            # Here's where we'd try something new to fix it. 
            best_match, confidence, text = rotateImage(data, json_data)
            num_spec = sum(v for k, v in ct.Counter(text).items() if k in special_chars)
        # if (len(text) < 4 or num_spec > 3 or confidence < 0.6):
        #     # try something else
        #     print(text)
        
        # Save to file
        end = {
            'File':             os.listdir(ALL_DIR)[i],
            'Actual_Title':     actual_title,
            'OCR':              text,
            'Predicted_Title':  best_match,
            'Confidence':       confidence,
            # 'Dict_Entry':       json_data.get(best_match), 
            # can pollute the json real bad, may help with debug
        }
        file = open(out_file, 'a')
        json.dump(end, file)
        file.write('\n')
        return end
    

def statistics(guesses: dict):
    accuracy = 0
    for entry in guesses.keys():
        entry = guesses.get(entry)
        if (entry.get('Actual_Title') == None or
            entry.get('Predicted_Title') == None):
            continue
        cleaned_predict = re.sub(r'[^\w\s]', '', entry.get('Predicted_Title'))
        cleaned_predict = re.sub(r'\s+', ' ', cleaned_predict).strip()
        cleaned_actual = re.sub(r'[^\w\s]', '', entry.get('Actual_Title'))
        cleaned_actual = re.sub(r'\s+', ' ', cleaned_actual).strip()
        # remove some special characters because /certain cards/ were being annoying
        if (entry.get('Actual_Title').lower() == entry.get('Predicted_Title').lower() or 
            cleaned_actual.lower() == cleaned_predict.lower()):
            accuracy += 1
    accuracy /= len(guesses.keys())
    return accuracy


if __name__ == '__main__':
    start_time = time.time()
    json_data = convert_json_file('data/full_data.json')
    
    # Clean output file for this run
    # shutil.rmtree(img_folder)
    # os.mkdir(img_folder)
    if os.path.exists(out_file):
        os.remove(out_file)

    out_dict = {}
    OCR = SimpleOCR()
    for i in range(0, num_imgs):
        response = recognize(i, json_data, OCR)
        out_dict[response.get('File')] = response

    print(f'Accuracy: {statistics(out_dict)}')

    # IF we make it to the end, replace the list of dicts 
    # with a single dict for easier import/processing
    file = open(out_file, 'w')
    json.dump(out_dict, file)

    # Clean up on the way out
    print(f'Execution time: {time.time() - start_time} seconds')
    