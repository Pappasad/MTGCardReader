import os
import pandas as pd
import json
from APutil import listdir, ALL_DIR, CardDatastore
import numpy as np

full_data_path = os.path.join('data', 'full_data.json')

ROTATED = {'split', 'planar', 'battle', 'aftermath', 'adventure'}

class MTGRotationDatabase:

    _ROT_PATH = os.path.join('data', 'datasets', 'master_data.csv')

    def __init__(self, load=True):
        if load and os.path.exists(self._ROT_PATH):
            self.df = pd.read_csv(self._ROT_PATH, low_memory=False)
        else:
            with open(full_data_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            self.df = pd.DataFrame(data)

            #imgs = CardDatastore(listdir(ALL_DIR), save='all')
            imgs = CardDatastore.load('all')
            labels = imgs.labels
            paths = imgs.paths
            to_keep = []
            paths_to_add = []
            for i, row in self.df.iterrows():
                nameify = row['name'].replace('/', '_').replace(' ', '_')
                set = row['set']
                id = row['collector_number']
                file = f'{nameify}_-_{set}_{id}.jpg'
                path = os.path.join(ALL_DIR, file)
                if path in paths:
                    to_keep.append(i)
                    paths_to_add.append(path)

                print(i)
                if not isinstance(row['type_line'], float):
                    if 'Aftermath' in row['keywords']:
                        row['rotated'] = True
                    elif 'battle' in row['type_line'].lower():
                        row['rotated'] = True

            self.df = self.df.iloc[to_keep].reset_index(drop=True)
            self.df['file_name'] = paths_to_add

            #print(self.df)
            self.df['rotated'] = self.df['layout'].isin(ROTATED)
            conditions = [
                self.df['keywords'].str.contains('Aftermath', na=False),
                self.df['layout'] == 'split',
                self.df['layout'] == 'adventure',
                self.df['rotated']
            ]
            choices = ['aftermath', 'split', 'adventure', 'rotated']
            self.df['rot_type'] = np.select(conditions, choices, default='none')

            shadows = lambda row: os.path.join('data', 'images', 'varied_shadows', os.path.splitext(os.path.basename(row['file_name']))[0] + '.png')
            self.df['shadow'] = self.df.apply(shadows, axis=1)
            self.df = self.df[[os.path.exists(path) for path in self.df['shadow']]]

            self.df.to_csv(self._ROT_PATH, index=False)

        # self.df['rotated'] = self.df['layout'].isin(ROTATED)
        # conditions = [
        #     self.df['keywords'].str.contains('Aftermath', na=False),
        #     self.df['layout'] == 'split',
        #     self.df['layout'] == 'adventure',
        #     self.df['rotated']
        # ]
        # choices = ['aftermath', 'split', 'adventure', 'rotated']
        # self.df['rot_type'] = np.select(conditions, choices, default='none')


        self.rotations = self.df[self.df['rotated']].reset_index(drop=True)
        self.non_rotations = self.df[~self.df['rotated']].reset_index(drop=True)
       # self.df.to_csv(self._ROT_PATH, index=False)

    def __getitem__(self, idx):
        if idx in list(self.df.columns):
            return self.df[idx]
        else:
            return self.df.iloc[idx]

    def __iter__(self):
        return self.df.iterrows()
    
    def __contains__(self, item):
        return item in self.df['name']
    


class MTGBalancedDatabase:
    __BALANCED_PATH = os.path.join(os.path.dirname(__file__), 'balanced_dataset.csv')

    def __init__(self, **kwargs):
        from_scratch = kwargs.get('from_scratch')
        if not from_scratch and os.path.exists(self.__BALANCED_PATH):
            self.df = pd.read_csv(self.__BALANCED_PATH, low_memory=False)
            # self.df = self.df.loc[self.df['file_name'] != 'rotated']
            # self.df.to_csv(self.__BALANCED_PATH, index=False)
        else:
            df = pd.read_csv(MTGRotationDatabase._ROT_PATH, low_memory=False)
            showcase = df[
                df['frame_effects'].astype(str).apply(lambda x: 'showcase' in x) |
                df['promo_types'].astype(str).apply(lambda x: 'showcase' in x)
            ]
            set_counts = showcase['set'].value_counts()
            limit = set_counts[set_counts >= 3].index
            showcase = showcase.loc[showcase['set'].isin(limit)].copy()
            rotated = df.loc[df['rotated'] == True].copy()
           
            normal = df.merge(rotated, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
            normal = normal.merge(showcase, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
            normal['img_type'] = 'normal'
            showcase['img_type'] = 'showcase'
            rotated.loc['img_type'] = 'rotated'

            data_min = min(len(showcase), rotated['rot_type'].value_counts().min())

            seed = kwargs.get('seed', 0)

            normal = normal.sample(n=data_min, random_state=seed)
            showcase = showcase.sample(n=data_min, random_state=seed)
            rotated = rotated.groupby('rot_type')

            self.df = pd.concat([normal, showcase], axis=0)
            for type, r in rotated:
                print(type)
                self.df = pd.concat([self.df, r], axis=0)
            
            self.df.to_csv(self.__BALANCED_PATH, index=False)











if __name__ == '__main__':
    db = MTGRotationDatabase(load=True)
    # print(db.rotations)
    db = MTGBalancedDatabase(from_scratch=True)