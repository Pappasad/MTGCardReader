import os
import numpy as np
import subprocess
from APnetwork import MTGRotationNetwork, MTGRotClassificationNetwork
from APPaddleOcr import NewOCR
from PIL import Image
from APdatabase import MTGRotationDatabase, MTGBalancedDatabase
import pandas as pd
import re
from KCMTGClassifier import classify_image

rem_dig_pattern = re.compile(r"\d+")
rem_dig = lambda s: rem_dig_pattern.sub('', s)
rem_dig = np.frompyfunc(rem_dig, 1, 1)


class CardReader:
    """
    CardReader class is the main class responsible for detecting card orientation, handling rotations,
    and performing OCR (Optical Character Recognition) on Magic: The Gathering (MTG) cards as well as return set, color, and rarity.
    """

    def __init__(self, hinton=False):
        """
        Initializes the CardReader instance.
        
        Args:
            hinton (bool): Whether to use batch processing and save images for later inference.
        """
        self.isRotated = MTGRotationNetwork.pretrained()  # Model to check if the card is rotated.
        self.getRotType = MTGRotClassificationNetwork.pretrained()  # Model to classify the rotation type.
        self.ocr = NewOCR(gpu=False)  # OCR model to extract text from images.
        self.hinton = hinton  # Flag to determine batch processing mode.
        self.mtgclassifier = classify_image

    def __call__(self, path: str | list | Image.Image, **kwargs):
        """
        Processes a given image path or a list of paths, detects card orientation,
        and applies OCR.
        
        Args:
            path (str | list): Path to a single image file or a list of paths.
            **kwargs: Additional arguments for handling batch processing.
        
        Returns:
            str | list: OCR results for single or multiple images.
        """
        if isinstance(path, list):  # Handle batch processing if a list is provided
            if self.hinton:
                results = np.empty(len(path), dtype='<U128')
                images = {'normal': [], 'rotated': [], 'aftermath': [], 'split': [], 'adventure': []}
                if not 'load' in kwargs:
                    for i, p in enumerate(path):
                        self(p, images=images, index=i)

                    # Convert images to numpy arrays
                    normal = np.array(images['normal'], dtype=object)
                    rotated = np.array(images['rotated'], dtype=object)
                    aftermaths = np.array(images['aftermath'], dtype=object)
                    split = np.array(images['split'], dtype=object)
                    adventures = np.array(images['adventure'], dtype=object)

                    # Save processed images for external script processing
                    single_input = np.concatenate([normal, rotated])
                    double_input = np.concatenate([aftermaths, split, adventures])

                    if 'runNsave' in kwargs:
                        os.makedirs('temp_data', exist_ok=True)
                        single = np.array_split(single_input, kwargs['runNsave'], axis=0)
                        for i, s in enumerate(single):
                            np.save(os.path.join('temp_data', f'single{i}.npy'), s)
                        single_path = 'temp_data'
                    else:
                        np.save('card-reader-single-inputs.npy', single_input)
                        single_path = 'card-reader-single-inputs.npy'

                    np.save('card-reader-double-inputs.npy', double_input)

                # Run OCR script on saved images
                subprocess.run(['python', os.path.join('code', 'cr-ocr-script.py'), single_path, 'single-input-outputs.npy'])
                subprocess.run(['python', os.path.join('code', 'cr-ocr-script.py'), 'card-reader-double-inputs.npy', 'double-input-outputs.npy'])

                outputs = np.load('single-input-outputs.npy', allow_pickle=True)
                outputs = np.concatenate([outputs, np.load('double-input-outputs.npy', allow_pickle=True)], axis=0)
                for item in outputs:
                    idx, result = item
                    results[int(idx)] = result

                return results         
            else:
                preds = []
                for i, p in enumerate(path):
                    preds.append(self(p))
                return preds
        elif isinstance(path, str):
            image = Image.open(path).convert('RGB')
        else:
            image = path
            
        bg = self.mtgclassifier(path)
        if self.isRotated(image):  # Check if image is rotated
            rotation = self.getRotType(image)  # Get rotation type
           # print(rotation)
            if rotation == 'aftermath':  # Aftermath cards
                face1 = image.crop((0, 0, image.width, image.height//2))
                face2 = image.crop((0, image.height//2, image.width, image.height)).rotate(90, expand=True)
                if 'images' in kwargs:
                    kwargs['images']['aftermath'].append([kwargs['index'], np.array(face1), np.array(face2)])
                    return
                return self.ocr(face1) + ' // ' + self.ocr(face2), bg
            elif rotation == 'split':  # Split cards
                y = image.height-1
                color = image.getpixel((200, y))
                while np.sqrt(color[0]**2 + color[1]**2 + color[2]**2) < 50:
                    y -= 1
                    color = image.getpixel((200, y))
                image = image.crop((0, 0, image.width, y))
                face2 = image.crop((0, 0, image.width, image.height//2)).rotate(-90, expand=True)
                face1 = image.crop((0, image.height//2, image.width, image.height)).rotate(-90, expand=True)
                if 'images' in kwargs:
                    kwargs['images']['split'].append([kwargs['index'], np.array(face1), np.array(face2)])
                    return
                return self.ocr(face1) + ' // ' + self.ocr(face2), bg
            elif rotation == 'adventure': #adventure cards
                face1 = image.crop((0, 0, image.width, image.height//2))
                face2 = image.crop((0, 3.75*image.height//6, image.width//2, image.height))
                if 'images' in kwargs:
                    kwargs['images']['adventure'].append([kwargs['index'], np.array(face1), np.array(face2)])
                    return
                return self.ocr(face1) + ' // ' + self.ocr(face2), bg
            else:  # Rotated card
                image = image.rotate(-90, expand=True)
                if 'images' in kwargs:
                    kwargs['images']['rotated'].append([kwargs['index'], np.array(image)])
                    return
                return self.ocr(image), bg
        else:  # Normal card
            if 'images' in kwargs:
                kwargs['images']['normal'].append([kwargs['index'], np.array(image)])
                return
            return self.ocr(image), bg


def cputest():
    """
    Runs a CPU-based test to evaluate OCR accuracy on a dataset of rotated and non-rotated MTG cards.
    """
    db = MTGRotationDatabase()
    nor = db.non_rotations.sample(n=len(db.rotations), random_state=0)

    N = 10
    afters = db.rotations.loc[db.rotations['rot_type'] == 'aftermath'].sample(n=N, random_state=0)
    splits = db.rotations.loc[db.rotations['rot_type'] == 'split'].sample(n=N, random_state=0)
    rotats = db.rotations.loc[db.rotations['rot_type'] == 'rotated'].sample(n=N, random_state=0)
    nor = nor.sample(n=N, random_state=0)
    rot = pd.concat([afters, splits, rotats], axis=0)

    all = pd.concat([rot, nor], axis=0)
    paths = list(all['file_name'])
    reader = CardReader()
    preds = reader(paths)
    y_hat_no_net = np.empty(len(preds), dtype='<U128')
    for i, path in enumerate(paths):
        y_hat_no_net[i] = reader.ocr(path)

    y = np.array(all['name'])
    y_hat = np.array(preds)

    accuracy = np.sum(y == y_hat) / len(y)
    def_accuracy = np.sum(y == y_hat_no_net) / len(y)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Default Accuracy: {def_accuracy*100:.2f}%")
    print(f"Improvement Factor: {accuracy/def_accuracy}")

def loadAndCheck(num):
    y = np.load('y.npy', allow_pickle=True)
    y_hat = np.load('y_hat.npy', allow_pickle=True)
    for i in range(2, num+1):
        print(i)
        y = np.concatenate([y, np.load(f'y{i}.npy', allow_pickle=True)])
        y_hat = np.concatenate([y_hat, np.load(f'y_hat{i}.npy', allow_pickle=True)])

    accuracy = np.sum(y == y_hat) / len(y)

    def postprocess(arr):
        arr = arr.astype(str)
        arr = np.char.replace(arr, ' ', '')
        arr = np.char.replace(arr, '.', '')
        arr = np.char.lower(arr)
        arr = rem_dig(arr)
        return arr
    
    yh_post = postprocess(y_hat)
    y_post = postprocess(y)

    postaccuracy = np.sum(yh_post == y_post) / len(y_post)

    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Real Accuracy: {postaccuracy*100:.2f}%")



if __name__ == '__main__':
    pass
    # db = pd.read_csv(os.path.join('code', 'superdata', 'OCR_NonShadow_analysis.csv'))
    # adventure = db.loc[db['rot_type'] == 'split'].iloc[:10].reset_index()
    # reader = CardReader()
    # for i, row in adventure.iterrows():
    #     print(f"{row['name']} -> {reader(row['file_name'])} -> {row['prediction']}: {row['file_name']}")
    # # loadAndCheck(4)
#     reader = CardReader(hinton=True)
#     db = MTGRotationDatabase()
#     paths1 = list(db['file_name'][10001:30000])
#     paths2 = list(db['file_name'][30001:60000])
#     paths3 = list(db['file_name'][60001:])
    
#     y_hat = reader(paths1, runNsave=40)
#     y = np.array(db['name'][10001:30000], dtype='<U128')
#     correct = y == y_hat
#    # print(f"Accuracy: {np.sum(correct)/len(y)*100:.2f}%")
#     # print("Incorrect:")
#     # for right, wrong in zip(y[y!=y_hat], y_hat[y != y_hat]):
#     #     print(right, wrong)

#     np.save('y_hat2.npy', y_hat)
#     np.save('y2.npy', y)

#     y_hat = reader(paths2, runNsave=60)
#     y = np.array(db['name'][30001:60000], dtype='<U128')
#     correct = y == y_hat

#     np.save('y_hat3.npy', y_hat)
#     np.save('y3.npy', y)

#     y_hat = reader(paths3, runNsave=80)
#     y = np.array(db['name'][60001:], dtype='<U128')
#     correct = y == y_hat

#     np.save('y_hat4.npy', y_hat)
#     np.save('y4.npy', y)
