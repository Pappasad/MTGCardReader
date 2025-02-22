import os
import code.APutil as pre
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

IMG_FOLDER = os.path.join('data', 'images', 'all')

def getBaselineStats():
    files = os.listdir(IMG_FOLDER)
    image_data = pd.DataFrame()
    image_data['Labels'] = [file[:file.find(pre.LABEL_FLAG)] for file in files]

    num_images = len(files)

    print(f'Total Images: {num_images}')

    num_classes = len(image_data['Labels'].value_counts())

    top10 = image_data['Labels'].value_counts().head(10)
    
    baseline_accuracy = top10.iloc[0] / num_images

    print(f"Num Classes: {num_classes}")
    print(f"Baseline Accuracy: {baseline_accuracy*100}%\n")
    print("Top 10:")
    idx = 1
    for label, counts in top10.items():
        print(f"#{idx}. {label} - {counts}")
        idx += 1


    #plt.imshow(mode_img)
    plt.axis('off')
    plt.show()

    
if __name__ == '__main__':
    getBaselineStats()
