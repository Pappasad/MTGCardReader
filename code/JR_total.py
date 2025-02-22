# Basic idea here: integrate Katie and Aidan's models

import json
import os
import re
import sys
import time

# from resnet import MTGClassifier
from code.APPaddleOcr import NewOCR
from fuzzywuzzy import fuzz


debug = 1                           # do you want nothing saved, the last step, or everything [0, 1, 2]
num_imgs = int(sys.argv[1])         # len(os.listdir(ALL_DIR)) # how many images to test
img_folder = 'test_img_storage/'    # save debug images in this folder
out_file = f'{img_folder}output.json'
regex = re.compile('[^a-zA-Z]')
ALL_DIR = os.path.join('data', 'images', 'all')

def convert_json_file(file: str) -> dict:
    json_file = open(file, 'r')
    json_data = json.load(json_file)
    diction = {}
    for item in json_data:
        if item.get('lang') != 'en':
            continue
        diction[item.get('name').lower() + "_-_" + str(item.get('mtgo_id')) + "_" + str(item.get('tcgplayer_id'))] = item
    return diction

def find_best_match(json_data, target_string):
    best_match = None
    best_score = 0
    clean_target = regex.sub('', target_string).replace(" ", "").strip().lower()
    for key in json_data.keys():
        key = key[:key.rfind('_-_')]
        clean_key = regex.sub('', key).replace(" ", "").strip().lower()
        score = fuzz.ratio(clean_key, clean_target)
        if score > best_score:
            best_match = key
            best_score = score
    return best_match, best_score

def find_best_key(json_data, target_string):
    best_score = 0
    out_key = None
    clean_target = regex.sub('', target_string).replace(" ", "").strip().lower()
    for key in json_data.keys():
        org_key = key[:key.rfind('_-_')]
        clean_key = regex.sub('', org_key).replace(" ", "").strip().lower()
        score = fuzz.ratio(clean_key, clean_target)
        if score > best_score:
            best_score = score
            out_key = key
    return out_key

def recognize(i: int, json_data: dict, OCR: NewOCR, title="") -> dict:
    # Get actual card title
    if (title != ""):
        actual_title = title[title.rfind('/') + 1:title.rfind('_-_')].replace('_', ' ').strip()
        file = title
    else:
        actual_title = os.listdir(ALL_DIR)[i]
        actual_title = actual_title[:actual_title.rfind('_-_')].replace('_', ' ').strip()
        file = os.path.join(ALL_DIR, os.listdir(ALL_DIR)[i])
    print(actual_title)

    # OCR
    org_text = OCR(file) #, returnAll=True)
    
    # Check for blank cards
    if (org_text.replace(" ", "") == "" or org_text == None):
        best_match = "Blank Card"
        best_match, confidence = find_best_match(json_data, best_match.lower())
    else:
        best_match, confidence = find_best_match(json_data, org_text.lower())
    
    # Get other data
    all_data = json_data.get(actual_title)
    if (all_data == None):
        all_title = find_best_key(json_data, actual_title.lower())
        all_data = json_data.get(all_title)
    # print(all_data)

    # Save to file
    end = {
        'File':             file,
        'Actual_Title':     actual_title,
        'OCR':              org_text,
        'Predicted_Title':  best_match,
        'Confidence':       confidence,
    }
    if (all_data != None):
        end['actual_set'] = all_data.get('set')
        end['actual_set_name'] = all_data.get('set_name')
        end['actual_colors'] = all_data.get('colors')
        end['actual_rarity'] = all_data.get('rarity')
    return end

def statistics(guesses: dict):
    accuracy_title  = 0
    accuracy_rarity = 0
    accuracy_color  = 0
    accuracy_set    = 0
    for entry in guesses.keys():
        entry = guesses.get(entry)
        # Title
        if (entry.get('Actual_Title') != None and
            entry.get('Predicted_Title') != None):
            cleaned_predict = regex.sub('', entry.get('Predicted_Title')).replace(" ", "").strip().lower()
            cleaned_actual = regex.sub('', entry.get('Actual_Title')).replace(" ", "").strip().lower()
            # remove some special characters and spaces because /certain cards/ were being annoying
            if (entry.get('Actual_Title').lower() == entry.get('Predicted_Title').lower() or 
                cleaned_actual.lower() == cleaned_predict.lower()):
                accuracy_title += 1
        # rarity
        # color
        # set
    accuracy_title  /= len(guesses.keys())
    accuracy_rarity /= len(guesses.keys())
    accuracy_color  /= len(guesses.keys())
    accuracy_set    /= len(guesses.keys())
    return [accuracy_title, accuracy_rarity, accuracy_color, accuracy_set]

if __name__ == '__main__':
    start_time = time.time()
    json_data = convert_json_file('data/full_data.json')
    OCR = NewOCR(gpu=True)
    # OtherCNN = MTGClassifier()

    # Clean output file for this run
    if os.path.exists(out_file):
        os.remove(out_file)

    try:
        test_file = sys.argv[2]
        print(test_file)
        out_dict = recognize(1, json_data, OCR, title=test_file)
    except:
        print("Doing Multiple")
        out_dict = {}
        for i in range(0, num_imgs): # 0, num_imgs # 5831
            response = recognize(i, json_data, OCR)
            out_dict[response.get('File')] = response

    if ('OCR' in out_dict.keys()):
        print(out_dict)
    else:
        accuracies = statistics(out_dict)
        print(f'Title  Accuracy: {accuracies[0]}')
        print(f'Rarity Accuracy: {accuracies[1]}')
        print(f'Color  Accuracy: {accuracies[2]}')
        print(f'Set    Accuracy: {accuracies[3]}')

    # IF we make it to the end, replace the list of dicts 
    # with a single dict for easier import/processing
    file = open(out_file, 'w')
    json.dump(out_dict, file)

    # Clean up on the way out

    print(f'Execution time: {time.time() - start_time} seconds')