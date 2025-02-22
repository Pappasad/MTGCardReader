from cardreader import CardReader
from database import MTGBalancedDatabase
import numpy as np
import os
from code.APEasyOcr import CharacterErrorRate, SimpleOCR
from code.APPaddleOcr import NewOCR
import re
from textwrap import dedent
from subprocess import run as runScript
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

directory  = os.path.join(os.path.dirname(__file__), 'superdata')
os.makedirs(directory, exist_ok=True)

rem_dig_pattern = re.compile(r"\d+")
rem_dig = lambda s: rem_dig_pattern.sub('', s)
rem_dig = np.frompyfunc(rem_dig, 1, 1)

def fixString(arr: np.array):
    arr = arr.astype(str)
    arr = np.char.replace(arr, ' ', '')
    arr = np.char.replace(arr, '.', '')
    arr = np.char.lower(arr)
    arr = rem_dig(arr)
    return arr


def createMetricsOCR(db: MTGBalancedDatabase, name: str):
    names = db.df['name']
    paths = list(db.df['file_name'])

    cers = np.zeros(4, dtype=np.float32)

    print("Baseline... ", end=' ')
    baseline_acc = names.value_counts().max() / len(names) * 100
    print(f"{baseline_acc:.2f}%")
    y = fixString(np.array(list(names), dtype='<U128'))
    y_hat = np.full(len(y), names.value_counts().idxmax())

    df, cer = CharacterErrorRate(y, y_hat)
    print(cer)
    cers[0] = cer*100

    print("Loading Simple Model...")
    simpleModel = SimpleOCR(fast=True)
    simple_func = lambda row: simpleModel(row['file_name'])
    def simple_func(row):
        return simpleModel(row['file_name'])
    print("Simple... ", end=' ')
    simple_predictions = fixString(np.array(list(db.df.apply(simple_func, axis=1)), dtype='<U128'))
    simple_acc = np.sum(simple_predictions == y) / len(y) * 100
    print(f"{simple_acc:.2f}%")

    df, cer = CharacterErrorRate(y, simple_predictions)
    cers[1] = cer*100

    # simple = db.df.copy()
    # simple['Correct'] = simple_predictions == y
    # simple['Pred'] = simple_predictions
    
    print("Creating adv script...")
    temp_path = os.path.join(os.path.dirname(__file__), 'temp')
    script_path = temp_path+'.py'
    np.save(temp_path+'.npy', np.array(paths, dtype='<U512'))
    temp_script = f'''
        import numpy as np
        from newocr import NewOCR

        paths = np.load("{temp_path}.npy")
        results = np.empty(len(paths), dtype="<U128")
        model = NewOCR(gpu=True)

        for idx, path in enumerate(paths):
            results[idx] = model(path)

        np.save("{temp_path}.npy", results)
    '''
    temp_script = dedent(temp_script)
    with open(script_path, 'w') as f:
        f.write(temp_script)
    print('Advanced... ', end=' ')
    runScript(['python', script_path])
    adv_predictions = fixString(np.load(temp_path+'.npy'))
    adv_acc = np.sum(adv_predictions == y) / len(y) * 100
    print(f"{adv_acc:.2f}%")
    os.remove(script_path)

    df, cer = CharacterErrorRate(y, adv_predictions)
    cers[2] = cer*100

    # adv = db.df.copy()
    # adv['Correct'] = adv_predictions == y
    # adv['Pred'] = adv_predictions

    print("Creating cardreader model...")
    model = CardReader(hinton=True)
    print("Rotation Adjusted... ", end=' ')
    rot_adj_predictions = fixString(model(paths))
    rot_adj_acc = np.sum(rot_adj_predictions == y) / len(y) * 100
    print(f"{rot_adj_acc:.2f}%")

    df, cer = CharacterErrorRate(y, rot_adj_predictions)
    cers[3] = cer*100

    analysis = db.df.copy()
    analysis['correct'] = rot_adj_predictions == y
    analysis['prediction'] = rot_adj_predictions

    acc = np.array([baseline_acc, simple_acc, adv_acc, rot_adj_acc], dtype=np.float32)
    save_path = os.path.join(directory, name+'_acc.npy')
    np.save(save_path, acc)
    # results = np.array([
    #     y == names.value_counts().idxmax(),
    #     y == simple_predictions,
    #     y == adv_predictions,
    #     y == rot_adj_predictions
    # ])
    # save_path = os.path.join(directory, name+'_results.npy')
    # np.save(save_path, results)
    save_path = os.path.join(directory, name+'_analysis.csv')
    analysis.to_csv(save_path, index=False)
    print("Saved", name)
    for cer in cers:
        print(f"{cer:.2f}%")
    np.save('cers.npy', cers)

def runNonShadow():
    db = MTGBalancedDatabase()
    createMetricsOCR(db, 'OCR_NonShadow')
    print("All done.")

def runShadow():
    db = MTGBalancedDatabase()
    paths = list(db.df['shadow'])
    y = fixString(np.array(db.df['name']))

    print("Creating cardreader model...")
    model = CardReader(hinton=True)
    print("Rotation Adjusted... ", end=' ')
    rot_adj_predictions = fixString(model(paths))
    rot_adj_acc = np.sum(rot_adj_predictions == y) / len(y) * 100
    print(f"{rot_adj_acc:.2f}%")

    df, cer = CharacterErrorRate(y, rot_adj_predictions)
    cer = cer*100

    print(cer)

def corr():
    path = os.path.join(directory, 'OCR_NonShadow_analysis.csv')
    analysis = pd.read_csv(path)

    X = analysis.drop(['prediction', 'correct'], axis=1)
    y = analysis['correct']

    encoder = LabelEncoder()
    numeric = analysis.select_dtypes(include=[np.number])
    categorical = analysis.select_dtypes(include=['object'])
    categorical = pd.get_dummies(categorical, drop_first=True)
    print(analysis.dtypes)
    exit()

    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances.sort_values(ascending=False)
    print(importances)

def stats():
    path = os.path.join(directory, 'OCR_NonShadow_analysis.csv')
    analysis = pd.read_csv(path)
    # rows2keep = []
    # for i, row in analysis.iterrows():
    #     print(row['prediction'])
    #     if str(row['prediction']) and str(row['prediction']).count('//') <= 1:
    #         rows2keep.append(i)
    # analysis = analysis.loc[rows2keep]
    # analysis.to_csv(path, index=False)
    mistakes = analysis.loc[~analysis['correct']]

    plt.figure(figsize=(8, 5))
    x = mistakes['frame'].unique()
    y = mistakes['frame'].value_counts()
    plt.bar(x, y, color='Red')
    plt.xlabel('Frame Year')
    plt.ylabel('Num Mistakes')
    plt.title('Num of Mistakes By Frame Year')
    plt.savefig(os.path.join('plots', 'frame.png'))

    plt.figure(figsize=(8, 5))
    x = mistakes['rot_type'].unique()
    y = mistakes['rot_type'].value_counts()
    plt.bar(x, y, color='Red')
    plt.xlabel('Card Layout')
    plt.ylabel('Num Mistakes')
    plt.title('Num of Mistakes By Card Orientation')
    plt.savefig(os.path.join('plots', 'orientation.png'))

    plt.figure(figsize=(8, 5))
    x = mistakes['produced_mana'].unique()
    y = mistakes['produced_mana'].value_counts()
    plt.bar(x, y, color='Red')
    plt.xlabel('Card Color')
    plt.ylabel('Num Mistakes')
    plt.title('Num of Mistakes By Card Color')
    plt.gca().yaxis.set_tick_params(rotation=90)
    plt.savefig(os.path.join('plots', 'color.png'))



def testy():
    df = pd.read_csv(os.path.join(directory, 'OCR_NonShadow_analysis.csv'))
    df, cer = CharacterErrorRate(fixString(np.array(df['name'])), df['prediction'])
    print(cer*100)


if __name__ == '__main__':
    #stats()
    #runShadow()
    corr()

    # db = MTGBalancedDatabase()
    # model = SimpleOCR(fast=True)
    # path = list(db.df['file_name'])[0]
    # print(model(path))
        

