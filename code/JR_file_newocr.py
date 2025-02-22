import re, time, sys, json
from code.APPaddleOcr import NewOCR
from fuzzywuzzy import fuzz

img_test = sys.argv[1]        # len(os.listdir(ALL_DIR)) # how many images to test
regex = re.compile('[^a-zA-Z]')


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
    diction = {}
    for item in json_data:
        if item.get('lang') != 'en':
            continue
        diction[item.get('name').lower()] = item
    return diction


def recognize(json_data: dict, OCR: NewOCR):
        # Get actual card title
        actual_title = img_test[img_test.rfind('/') + 1:img_test.rfind('_-_')].replace('_', ' ').strip()
        
        # OCR
        org_text = OCR(img_test)
        # print(org_text)

        # Check for blank cards
        if (org_text.replace(" ", "") == "" or org_text == None):
            # print(file)
            best_match = "Blank Card"
            best_match, confidence = find_best_match(json_data, best_match.lower())
        else:
            best_match, confidence = find_best_match(json_data, org_text.lower())
        
        # Save to file
        end = {
            'File':             img_test,
            'Actual_Title':     actual_title,
            'Predicted_Title':  best_match,
            'Confidence':       confidence,
        }
        print(end)


if __name__ == '__main__':
    start_time = time.time()
    json_data = convert_json_file('data/full_data.json')

    OCR = NewOCR(gpu=True)
    response = recognize(json_data, OCR)

    # Clean up on the way out
    print(f'Execution time: {time.time() - start_time} seconds')
    