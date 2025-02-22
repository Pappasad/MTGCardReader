from APcardreader import CardReader
import os
import sys
import numpy as np
import subprocess
from APnetwork import MTGRotationNetwork, MTGRotClassificationNetwork
from APPaddleOcr import NewOCR
from textwrap import dedent
from PIL import Image
from APdatabase import MTGRotationDatabase, MTGBalancedDatabase
import re
from APEasyOcr import CharacterErrorRate

rem_dig_pattern = re.compile(r"\d+")
rem_dig = lambda s: rem_dig_pattern.sub('', s)
rem_dig = np.frompyfunc(rem_dig, 1, 1)

def loadAndCheck(num):
    y = np.load('y.npy', allow_pickle=True)
    y_hat = np.load('y_hat.npy', allow_pickle=True)
    for i in range(2, num+1):
        print(i)
        y = np.concatenate([y, np.load(f'y{i}shadows.npy', allow_pickle=True)])
        y_hat = np.concatenate([y_hat, np.load(f'y_hat{i}shadows.npy', allow_pickle=True)])

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
    df,cer = CharacterErrorRate(y_post, yh_post)
    print(cer*100)


if __name__ == '__main__':
    # reader = CardReader(hinton=False)
    # db = MTGBalancedDatabase()
    # paths1 = list(db.df['shadow'])
    # paths2 = list(db.df['shadow'])
    # paths3 = list(db.df['shadow'])
    
    # y_hat = reader(paths1, runNsave=40)
    # y = np.array(db.df['name'], dtype='<U128')
    # correct = y == y_hat
    loadAndCheck(4)

   # print(f"Accuracy: {np.sum(correct)/len(y)*100:.2f}%")
    # print("Incorrect:")
    # for right, wrong in zip(y[y!=y_hat], y_hat[y != y_hat]):
    #     print(right, wrong)

    # np.save('y_hat2shadows.npy', y_hat)
    # np.save('y2shadows.npy', y)

    # y_hat = reader(paths2, runNsave=60)
    # y = np.array(db.df['name'], dtype='<U128')
    # correct = y == y_hat

    # np.save('y_hat3shadows.npy', y_hat)
    # np.save('y3shadows.npy', y)

    # y_hat = reader(paths3, runNsave=80)
    # y = np.array(db.df['name'], dtype='<U128')
    # correct = y == y_hat

    # np.save('y_hat4shadows.npy', y_hat)
    # np.save('y4shadows.npy', y)
