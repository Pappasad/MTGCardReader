import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from APdatabase import MTGRotationDatabase
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pandas as pd

# # Set options to display all rows and columns
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

class CardDataSet(Dataset):
    def __init__(self, df):
        self.df = df
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path, label = row['file_name'], int(row['rotated'])

        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label
    
class RotClassDataset(CardDataSet):
    mapping = {"none": 0, "aftermath": 1, "split": 2, 'adventure': 3, "rotated": 4}

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path, label = row['file_name'], self.mapping[row['rot_type']]

        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label
    
class MTGRotationNetwork:
    _trained_path = os.path.join('models', 'mtgrot_trained.pth')

    @classmethod
    def pretrained(cls):
        if not os.path.exists(cls._trained_path):
            print("ERROR: No trained .pth")
            raise FileNotFoundError
        
        ins = cls.__new__(cls)
        ins.__init__()
        states = torch.load(cls._trained_path, map_location=ins.device)
        ins.model.load_state_dict(states)
        ins.model.eval()
        return ins

    def __init__(self):
        self.model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
        self.num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(self.num_features, 2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train: CardDataSet, valid: CardDataSet, epochs: int, lr=0.0001, batch_size=32, save=True):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        self.validate(valid_dataloader)
        if save:
            torch.save(self.model.state_dict(), self._trained_path)

    def validate(self, valid_dataloader: DataLoader):
        self.model.eval()
        correct = 0
        total = 0
        valid_loss = 0.0

        with torch.no_grad():
            for images, labels in valid_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                valid_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        valid_accuracy = 100 * correct / total
        print(f"Validation Loss: {valid_loss/len(valid_dataloader):.4f} Validation Acc: {valid_accuracy:.2f}%")

    def __call__(self, path: str | Image.Image):
        self.model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if isinstance(path, str):
            img = Image.open(path).convert('RGB')
        else:
            img = path
        img = transform(img)
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            _, predicted = torch.max(outputs, 1)

        return predicted.item()

class MTGRotClassificationNetwork(MTGRotationNetwork):
    _trained_path = os.path.join('models', 'mtgrotclass_trained.pth')

    def __init__(self):
        # Load EfficientNet B7 pre-trained model.
        self.model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
        self.num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(self.num_features, 5)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def __call__(self, path: str | Image.Image):
        num = super().__call__(path)
        return list(RotClassDataset.mapping.keys())[num]


def trainRotClass(db: MTGRotationDatabase):
    rotated = db.rotations
    number_of_items = db.rotations['rot_type'].value_counts().min()
    none = db.non_rotations.sample(n=number_of_items, random_state=0)
    split = rotated.loc[rotated['rot_type'] == 'split'].sample(n=number_of_items, random_state=0)
    rotate = rotated.loc[rotated['rot_type'] == 'rotated'].sample(n=number_of_items, random_state=0)
    aftermaths = rotated.loc[rotated['rot_type'] == 'aftermath'].sample(n=number_of_items, random_state=0)
    adventures = rotated.loc[rotated['rot_type'] == 'adventure'].sample(n=number_of_items, random_state=0)
    
    all = pd.concat([none, split, rotate, aftermaths, adventures], axis=0)

    shadow_data = all.copy()
    shadow_data['file_name'] = shadow_data['shadow']

    all = pd.concat([all, shadow_data], axis=0)

    tr, tstvld = train_test_split(all, test_size=0.2, random_state=0)
    vld, tst = train_test_split(tstvld, test_size=0.5, random_state=0)

    train = RotClassDataset(tr)
    valid = RotClassDataset(vld)

    network = MTGRotClassificationNetwork()
    network.train(train, valid, epochs=10)

    network = MTGRotClassificationNetwork.pretrained()

    y = np.array([RotClassDataset.mapping[rot] for rot in tst['rot_type']], dtype=int)
    y_hat = np.empty(len(y), dtype=int)
    for i, path in enumerate(tst['file_name']):
        #print(f"{y[i]} -> {network(path)}")
        y_hat[i] = list(RotClassDataset.mapping.keys()).index(network(path))
    correct = y == y_hat
    test_acc = np.sum(correct) / len(y)
    print(f"Rot Class Accuracy: {test_acc*100:.2f}%")
   # print(np.sum(y_hat == 3))

def trainRotation(db: MTGRotationDatabase):
    rotated = db.rotations
    non_rotated = db.non_rotations.sample(n=len(rotated), random_state=0)

    data = pd.concat([rotated, non_rotated], axis=0)

    shadow_data = data.copy()
    shadow_data['file_name'] = shadow_data['shadow']

    data = pd.concat([data, shadow_data], axis=0)

    tr, tstvld = train_test_split(data, test_size=0.2, random_state=0)
    vld, tst = train_test_split(tstvld, test_size=0.5, random_state=0)

    train = CardDataSet(tr)
    valid = CardDataSet(vld)

    network = MTGRotationNetwork()
    network.train(train, valid, epochs=10)
    network = MTGRotationNetwork.pretrained()

    y = np.array(tst['rotated'], dtype=bool)
    y_hat = np.empty(len(y), dtype=bool)
    for i, path in enumerate(tst['file_name']):
        print(network(path))
        y_hat[i] = network(path)
    correct = y == y_hat
    test_acc = np.sum(correct) / len(y)
    print(f"Rot Test Accuracy: {test_acc*100:.2f}%")
    

if __name__ == '__main__':
    db = MTGRotationDatabase(load=True)
    trainRotation(db)
    trainRotClass(db)
    