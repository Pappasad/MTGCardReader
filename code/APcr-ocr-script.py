import os
import sys
import numpy as np
from code.APPaddleOcr import NewOCR

if not len(sys.argv) == 3:
    print("BEWARE THAR BE DRAGONS HERE")
    sys.exit()

if os.path.isdir(sys.argv[1]):
    paths = os.listdir(sys.argv[1])
    inputs = np.load(os.path.join(sys.argv[1], paths[0]), allow_pickle=True)
    for path in paths[1:]:
        inputs = np.concatenate([inputs, np.load(os.path.join(sys.argv[1], path), allow_pickle=True)], axis=0)
else:
    inputs = np.load(sys.argv[1], allow_pickle=True)
outputs = np.empty(shape=(len(inputs), 2), dtype=object)

ocr = NewOCR(gpu=True)
for j, item in enumerate(inputs):
    try:
        if len(item) == 2:
            index, image = item
            outputs[j] = np.array([index, ocr(image)])
            #print(f"Predicting {index}")
        elif len(item) == 3:
            index, image1, image2 = item
            outputs[j] = np.array([index, ocr(image1) + ' // ' + ocr(image2)])
            #print(f"Predicting {index}")
        else:
            raise
    except Exception as e:
        print(f"Could not complete item for {sys.argv[1]}: {item.ndim} dimensions")
        raise e

np.save(sys.argv[2], outputs)
