import os
import torch
import cv2
import numpy as np
import easyocr
import evaluate
import pandas as pd
import Levenshtein
from code.APutil import ALL_DIR, listdir, path2Label, imread
import os

CER = evaluate.load('cer')

def CharacterErrorRate(targets, predictions):
    """
    Compute character error rate (CER) and return a DataFrame tracking each character edit operation.

    Args:
        targets (list of str): List of correct reference texts.
        predictions (list of str): List of OCR model outputs.

    Returns:
        df (pd.DataFrame): Table of edit operations (Substitutions, Insertions, Deletions).
        cer (float): Overall Character Error Rate.
    """
    all_edits = []

    # Process each target-prediction pair
    for target_idx, (target, prediction) in enumerate(zip(targets, predictions)):
        edits = Levenshtein.editops(target, prediction)

        for op, target_pos, pred_pos in edits:
            original_char = target[target_pos] if op in ["replace", "delete"] else "-"
            predicted_char = prediction[pred_pos] if op in ["replace", "insert"] else "-"

            all_edits.append({
                "Target Index": target_idx,  # Which pair this belongs to
                "Operation": op,
                "Original": original_char,
                "Predicted": predicted_char
            })

    # Convert collected data into a Pandas DataFrame
    df = pd.DataFrame(all_edits, columns=["Target Index", "Operation", "Original", "Predicted"])

    # Compute CER using `evaluate`
    cer = CER.compute(predictions=predictions, references=targets)

    return df, cer           

class SimpleOCR:
    def __init__(self, fast=True):
        super().__init__()
        hasGPU = torch.cuda.is_available() and fast
        self.device = torch.device('cuda' if hasGPU else 'cpu')
        self.reader = easyocr.Reader(['en'], gpu=hasGPU, recog_network='english_g2', user_network_directory=None)
        
    def __call__(self, img: str) -> list[str]:
        try:
            img = imread(img)
            img = img.to(self.device)
            processed = self.threshold(img)
            pred = self.reader.readtext(processed, detail=0)
            return pred[0]
        except:
            return ' '
    
    def threshold(self, img: torch.tensor) -> np.ndarray:
        grayscale = 0.299*img[0] + 0.587*img[1] + 0.114*img[2]

        if grayscale.max() <= 1.0:
            grayscale *= 255.0

        grayscale = grayscale.to(torch.uint8).cpu().numpy()
        # Apply Otsu's thresholding
        threshold, binarized = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binarized
    

    

    


def testf():
    # img = 'test_image.png'
    # print('start')
    # result = NewOCR(img, returnAll=False)
    # print('finish')
    # print(result)

    # img = listdir(ALL_DIR)[:10]
    # print('start')
    # results = model.multiple(img)
    # print('finished')
    # for i, r in zip(img, results):
    #     print(f"{os.path.basename(i)} -> {r[0]}")



if __name__ == '__main__':
    testf()
