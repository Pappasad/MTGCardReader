import os, re, time, sys, json, string
from code.APPaddleOcr import NewOCR
from fuzzywuzzy import fuzz

debug = 1                           # do you want nothing saved, the last step, or everything [0, 1, 2]
num_imgs = int(sys.argv[1])         # len(os.listdir(ALL_DIR)) # how many images to test
img_folder = 'test_img_storage/'    # save debug images in this folder
CROP_DIR = os.path.join('data', 'images', 'crop') # save cropped images in this folder
special_chars = string.punctuation
out_file = f'{img_folder}out.json'
regex = re.compile('[^a-zA-Z]')
ALL_DIR = os.path.join('data', 'images', 'all')


def find_best_match(json_data, target_string):
    best_match = None
    best_score = 0
    clean_target = regex.sub('', target_string).replace(" ", "").strip().lower()
    for key in json_data.keys():
        clean_key = regex.sub('', key).replace(" ", "").strip().lower()
        score = fuzz.ratio(clean_key, clean_target)
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


def recognize(i: int, json_data: dict, OCR: NewOCR):
        # Get actual card title
        actual_title = os.listdir(ALL_DIR)[i]
        actual_title = actual_title[:actual_title.rfind('_-_')].replace('_', ' ').strip()

        # Get image
        file = os.path.join(ALL_DIR, os.listdir(ALL_DIR)[i])
        
        # OCR
        org_text = OCR(file) # data.numpy()

        # Check for blank cards
        if (org_text.replace(" ", "") == "" or org_text == None):
            # print(file)
            best_match = "Blank Card"
            best_match, confidence = find_best_match(json_data, best_match.lower())
        else:
            best_match, confidence = find_best_match(json_data, org_text.lower())
        
        # Save to file
        end = {
            'File':             os.listdir(ALL_DIR)[i],
            'Actual_Title':     actual_title,
            'OCR':              best_match,
            'Predicted_Title':  best_match,
            'Confidence':       confidence,
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
            continue # This was an issue with easyocr, less so with newocr
        cleaned_predict = regex.sub('', entry.get('Predicted_Title')).replace(" ", "").strip().lower()
        cleaned_actual = regex.sub('', entry.get('Actual_Title')).replace(" ", "").strip().lower()
        # remove some special characters and spaces because /certain cards/ were being annoying
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
    OCR = NewOCR(gpu=True)
    for i in range(0, num_imgs): # 0, num_imgs # 5831
        response = recognize(i, json_data, OCR)
        out_dict[response.get('File')] = response

    print(f'Accuracy: {statistics(out_dict)}')

    # IF we make it to the end, replace the list of dicts 
    # with a single dict for easier import/processing
    file = open(out_file, 'w')
    json.dump(out_dict, file)

    # Clean up on the way out
    print(f'Execution time: {time.time() - start_time} seconds')
    