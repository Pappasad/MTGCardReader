import os
import numpy as np
from sklearn.model_selection import train_test_split  # For splitting data into training, validation, and testing sets
from PIL import Image  # Library for image processing
import sys  # Module for interacting with the Python runtime environment
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import random

# Define paths for storing image data and metadata
ALL_DIR = os.path.join('data', 'images', 'all')
# Define path for augmented training images
AUGMENTED_TRAIN_DIR = os.path.join('data', 'images', 'augmented_train')
os.makedirs(AUGMENTED_TRAIN_DIR, exist_ok=True)

def listdir(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]

LABEL_FLAG = '_-_'
DEF_WIDTH = 488
DEF_HEIGHT = 680

def path2Label(path: str):
    file = os.path.basename(path)
    return file[:file.find(LABEL_FLAG)].replace('_', ' ')


# Define augmentation pipeline
augment_transform = transforms.Compose([
    transforms.RandomResizedCrop((DEF_HEIGHT, DEF_WIDTH), scale=(0.95, 1.0)), # Random zoom/crop (95%-100% of image)
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Offset image by up to 10% in x and y directions
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # Apply random Gaussian blur
    transforms.RandomRotation(degrees=5),  # Slight rotation (max 5 degrees)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # Adjust brightness & contrast
])


def augment_and_save_training_images(train_files, num_augmentations=10):
    # Applies augmentations and saves new images for the training set.
    augmented_files = []

    for img_path in train_files:
        original_img = Image.open(img_path).convert("RGB")
        img_name = os.path.basename(img_path).split('.')[0]  # Get image name

        # Save the original training image in augmented directory
        original_save_path = os.path.join(AUGMENTED_TRAIN_DIR, f"{img_name}_original.jpg")
        original_img.save(original_save_path)
        augmented_files.append(original_save_path)

        # Generate multiple augmentations for the training image
        for i in range(num_augmentations):
            augmented_img = augment_transform(original_img)  # Apply augmentation
            save_path = os.path.join(AUGMENTED_TRAIN_DIR, f"{img_name}_aug{i}.jpg")
            augmented_img.save(save_path)  # Save augmented image
            augmented_files.append(save_path)

    return augmented_files  # Return list of augmented training images


# Function to split the image data into training, validation, and testing sets
def trainTestSplit(train_size, test_size, valid_size, save=True, transform=None):
    # Normalize the test and validation sizes relative to their combined size
    test_size = test_size / (test_size + valid_size)
    valid_size = valid_size / (test_size + valid_size)

    # Get the list of image files in the directory
    img_files = [os.path.join(ALL_DIR, file) for file in os.listdir(ALL_DIR)[:10000]]

    # Split the data into training and test/validation sets
    train, test_and_valid = train_test_split(img_files, train_size=train_size, random_state=0)
    # Further split the test/validation set into separate test and validation sets
    valid, test = train_test_split(test_and_valid, test_size=test_size, random_state=0)

    if save:
        train = CardDatastore(train, save='train')
        test = CardDatastore(test, save='test')
        valid = CardDatastore(valid, save='valid')

    # **Augment only the training set**
    #augmented_train_files = augment_and_save_training_images(train, num_augmentations=10)  

    print(f"Original train size: {len(img_files) * train_size:.0f}")
    #print(f"Expanded train size: {len(augmented_train_files)}")

    print("All done splitting.")
    return train, test, valid

# Function to read an image and convert it to RGB format
def imread(path: str, remove=False) -> torch.tensor:
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((DEF_WIDTH, DEF_HEIGHT), Image.Resampling.LANCZOS)
        t = transforms.ToTensor()(img)
    except Exception as e:
        if remove:
            os.remove(path)
        print("IMREAD ERROR:", path)
        print('\n', e)
        sys.exit()
    return t

def imshow(t: torch.tensor, path='TEMP.png'):
    img = transforms.ToPILImage()(t)
    img.save(path)
    return True

# Class to manage card image data and metadata
class CardDatastore(Dataset):
    # Directory where the image files are stored
    _directory = ALL_DIR
    _data_dir = os.path.join('data', 'images')

    # Initialize with a list of image filenames
    def __init__(self, img_files, save='', transform=None):

        num_img, channels, height, width = len(img_files), *imread(img_files[0]).shape
        self._data = pd.DataFrame(columns=['file_name', 'text'], index=np.arange(num_img))

        print("Getting Data...")
        for idx, file in enumerate(img_files):
            self._data.iloc[idx] = {
                'file_name': file,
                'text': path2Label(file)
            }
            
        if save:
            print("Saving...")
            self.save(save)

        self.transform = transform

    # Save the data 
    def save(self, name: str):
        path = os.path.join(self._data_dir, name+'.csv')
        self._data.to_csv(path, index=False)
        print(f"Data saved to {path}")

    # Load the data
    @classmethod
    def load(cls, name: str):
        instance = cls.__new__(cls)
        instance.transform = None
        path = os.path.join(cls._data_dir, name+'.csv')
        instance._data = pd.read_csv(path)
        return instance

    # Get an item (image or label) by index or name
    def __getitem__(self, idx):
        path = self._data['Path'][idx]
        name = self._data['Label'][idx]
        num = self._data['Num'][idx]

        return path, name, num

    # Return the number of items in the datastore
    def __len__(self):
        return len(self.labels)
    
    # Check if an item exists in the datastore
    def __contains__(self, item):
        return item in self.data or item in self.labels or item in self.names
    
    def __iter__(self):
        return self._data.iterrows()
    
    @property
    def labels(self) -> np.array:
        return np.array(self._data['text'], dtype='<U64')
    
    @property
    def paths(self) -> np.array:
        return np.array(self._data['file_name'], dtype='<U256')
    
    def img(self, idx):
        return imread(self._data['file_name'][idx])
    



if __name__ == '__main__':
    train, test, valid = trainTestSplit(0.8, 0.1, 0.1)
    train = CardDatastore(train, save='train')
    test = CardDatastore(test, save='test')
    valid = CardDatastore(valid, save='valid')