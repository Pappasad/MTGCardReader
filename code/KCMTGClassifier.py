import os
import ast
import pandas as pd
from sklearn.model_selection import train_test_split
from APdatabase import MTGBalancedDatabase 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# DS for loading images and data
class MTGDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, rarity_map=None, set_map=None):
        # make text fields all lowercase, no spaces
        df["rarity"] = df["rarity"].astype(str).str.lower().str.strip()
        df["set"] = df["set"].astype(str).str.lower().str.strip()
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        # map rarity labels to int values
        self.rarity_map = rarity_map or {"common": 0, "uncommon": 1, "rare": 2, "mythic": 3}

    @staticmethod
    def colors_to_index(colors_list):
        # define bit for each color: W (white), U (blue), B (black), R (red), G (green)
        mapping = {"w": 1, "u": 2, "b": 4, "r": 8, "g": 16}
        index = 0
        # index for each color
        for color in colors_list:
            c = str(color).lower().strip()
            if c in mapping:
                index |= mapping[c] #bitwise or for color encoding
        return index

    def __len__(self):
        return len(self.df) # num samples

    def __getitem__(self, idx):
        # get row
        row = self.df.iloc[idx]
        raw_name = str(row["file_name"])
        prefix = "data/images/all/"
        if raw_name.startswith(prefix):
            raw_name = raw_name[len(prefix):] # remove prefix from file name
        file_name = os.path.basename(raw_name) # get actual file name
        img_path = os.path.join(self.img_dir, file_name) 

        image = Image.open(img_path).convert("RGB") # open image, make sure rgb
        if self.transform:
            image = self.transform(image)

        rarity_str = row["rarity"]
        rarity_label = self.rarity_map[rarity_str] # convert rarity str to int

        try:
            colors_list = ast.literal_eval(row["colors"])
        except Exception:
            colors_list = []
        # convert the full color list into a single index (0  =  colorless)
        color_label = self.colors_to_index(colors_list)

        set_str = row["set"]
        set_label = self.set_map[set_str] # convert set str to int

        # make target w tensors
        targets = {
            "rarity": torch.tensor(rarity_label, dtype=torch.long),
            "color": torch.tensor(color_label, dtype=torch.long),
            "set": torch.tensor(set_label, dtype=torch.long)
        }
        return image, targets

# CNN for classifying rarity, color, set
class MTGClassifier(nn.Module):
    def __init__(self, num_rarity, num_color, num_set):
        super(MTGClassifier, self).__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1) # use resnet as backbon
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity() # replace fc later w id layer (remove)
        self.backbone = backbone

        # add heads for each category
        self.rarity_head = nn.Linear(num_features, num_rarity)
        self.color_head = nn.Linear(num_features, num_color)
        self.set_head = nn.Linear(num_features, num_set)
        
    def forward(self, x):
        features = self.backbone(x) # extract features
        return {
            "rarity": self.rarity_head(features),
            "color": self.color_head(features),
            "set": self.set_head(features)
        }
    
    def classify(self, item):
        # classify w file path or a PIL Image.
        if isinstance(item, str):
            img = Image.open(item).convert("RGB")
        else:
            img = item
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        img = transform(img).unsqueeze(0).to(next(self.parameters()).device)
        self.eval()
        with torch.no_grad():
            outputs = self.forward(img)
        # get predicted class for each category
        rarity_idx = torch.argmax(outputs["rarity"]).item()
        color_idx = torch.argmax(outputs["color"]).item()
        set_idx = torch.argmax(outputs["set"]).item()
        return {
            "rarity": rarity_idx,
            "color": color_idx,
            "set": set_idx
        }

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total = 0
    for images, targets in dataloader:
        images = images.to(device)
        tr = targets["rarity"].to(device)
        tc = targets["color"].to(device)
        ts = targets["set"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = (criterion(outputs["rarity"], tr) +
                criterion(outputs["color"], tc) +
                criterion(outputs["set"], ts))
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size
    return running_loss / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct_rar = 0
    correct_col = 0
    correct_set = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            tr = targets["rarity"].to(device)
            tc = targets["color"].to(device)
            ts = targets["set"].to(device)

            set_loss_weight = 6.0  # Weight for set classification loss

            outputs = model(images)
            loss = (criterion(outputs["rarity"], tr) +
                    criterion(outputs["color"], tc) +
                    set_loss_weight * criterion(outputs["set"], ts))
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

            # accuracy for each category
            _, pr = torch.max(outputs["rarity"], 1)
            _, pc = torch.max(outputs["color"], 1)
            _, ps = torch.max(outputs["set"], 1)
            correct_rar += (pr == tr).sum().item()
            correct_col += (pc == tc).sum().item()
            correct_set += (ps == ts).sum().item()
    avg_loss = running_loss / total if total > 0 else 0.0
    acc_rar = correct_rar / total if total > 0 else 0.0
    acc_col = correct_col / total if total > 0 else 0.0
    acc_set = correct_set / total if total > 0 else 0.0
    return avg_loss, (acc_rar, acc_col, acc_set)

# get paths
code_dir = os.path.dirname(__file__)
csv_path = os.path.join('data', 'datasets', "master_data.csv")

# load data and make maps
df = pd.read_csv(csv_path, low_memory=False)
needed_cols = ["file_name", "rarity", "colors", "set"]
df = df.dropna(subset=needed_cols)
df.loc[:, "rarity"] = df["rarity"].astype(str).str.lower().str.strip()
df.loc[:, "set"] = df["set"].astype(str).str.lower().str.strip()
valid_rarities = {"common", "uncommon", "rare", "mythic"}
df = df[df["rarity"].isin(valid_rarities)]
df = df[df["set"] != ""]

global_set_map = {s: i for i, s in enumerate(df["set"].unique().tolist())}

# use reverse mapping for rarity and set mapping to get labels from ints
reverse_rarity_map = {0: "common", 1: "uncommon", 2: "rare", 3: "mythic"}
reverse_set_map = {v: k for k, v in global_set_map.items()}

# convert the color index back to a list of color abbreviations.
def color_index_to_labels(color_index):
    mapping = {"W": 1, "W": 2, "B": 4, "R": 8, "G": 16}
    labels = [c for c, bit in mapping.items() if color_index & bit]
    if not labels:
        labels.append("colorless")
    return labels


def classify_image(image_path):  
    # set model parameters
    num_rarity = 4
    num_color = 32
    num_set = len(global_set_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # go fast
    
    # init the model and load the last checkpoint
    model = MTGClassifier(num_rarity, num_color, num_set).to(device)
    ckpt_path = os.path.join('models', "best_all_color_model_test.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        raise FileNotFoundError("Model checkpoint not found!")
    
    # get predictions from the model using classify method.
    result = model.classify(image_path)
    
    result_labels = {
        "set": reverse_set_map[result["set"]],
        "color": color_index_to_labels(result["color"]),
        "rarity": reverse_rarity_map[result["rarity"]]
    }
    return result_labels


def main():
    # training parameters
    sample_size = None  # set to None for full dataset.
    num_epochs = 15
    batch_size = 32
    lr = 1e-4

    # get paths
    code_dir = os.path.dirname(__file__)
    csv_path = os.path.join(code_dir, "master_data.csv")
    project_root = os.path.dirname(code_dir)
    img_dir = os.path.join(project_root, "data", "images", "all")

    # get data
    df = pd.read_csv(csv_path, low_memory=False)
    needed_cols = ["file_name", "rarity", "colors", "set"]
    df = df.dropna(subset=needed_cols)

    # normalize (make all lowercase, no spaces)
    df["rarity"] = df["rarity"].astype(str).str.lower().str.strip()
    df["set"] = df["set"].astype(str).str.lower().str.strip()
    valid_rarities = {"common", "uncommon", "rare", "mythic"}
    df = df[df["rarity"].isin(valid_rarities)]
    df = df[df["set"] != ""]

    # make set map again
    global_set_map = {s: i for i, s in enumerate(df["set"].unique().tolist())}

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # set up dfs
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Dataset sizes: Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")

    # set up transformatoins
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # set up datasets
    train_ds = MTGDataset(train_df, img_dir, transform=transform, set_map=global_set_map)
    val_ds = MTGDataset(val_df, img_dir, transform=transform,
                        rarity_map=train_ds.rarity_map,
                        set_map=global_set_map)
    test_ds = MTGDataset(test_df, img_dir, transform=transform,
                         rarity_map=train_ds.rarity_map,
                         set_map=global_set_map)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    num_rarity = len(train_ds.rarity_map)
    num_color = 32  # 32 possible color combinations (2^5) using bits for [W, U, B, R, G]
    num_set = len(global_set_map)
    print("num_rarity:", num_rarity)
    print("num_color:", num_color)
    print("num_set:", num_set)

    # set up model w cross entropy loss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MTGClassifier(num_rarity, num_color, num_set).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ## TRAINING START - UNCOMMENT TO RETRAIN MODEL
    """
    best_val_loss = float("inf")
    for epoch in range(1, num_epochs+1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, (val_acc_rar, val_acc_col, val_acc_set) = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{num_epochs}  Train Loss={train_loss:.4f}  Val Loss={val_loss:.4f}")
        print(f"  Val Acc: Rarity={val_acc_rar*100:.1f}%  Color={val_acc_col*100:.1f}%  Set={val_acc_set*100:.1f}%")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(code_dir, "best_all_color_model_test.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  --> Saved best model to {ckpt_path}")
    """
    ## TRAINING END

    # #load the saved model checkpoint
    ckpt_path = os.path.join(code_dir, "best_all_color_model_test.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded model from {ckpt_path}")
    else:
        print("Checkpoint not found! Exiting evaluation.")
        return

    ## EVAL ON TEST SET START -  UNCOMMENT TO RUN MODEL ON TEST SET ONLY
    # # Evaluate the model on the test set:
    """
    test_loss, (test_acc_rar, test_acc_col, test_acc_set) = evaluate(model, test_loader, criterion, device)
    print("\nTest Set Evaluation:")
    print(f"  Test Loss={test_loss:.4f}")
    print(f"  Test Acc: Rarity={test_acc_rar*100:.1f}%  Color={test_acc_col*100:.1f}%  Set={test_acc_set*100:.1f}%")
    """
    # EVAL ON TEST SET END

    balanced_db = MTGBalancedDatabase(from_scratch=False)
    balanced_df = balanced_db.df.copy()

    # Filter out rows with rarities not in our valid set and rows whose 'set' values are not in the training mapping.
    valid_rarities = {"common", "uncommon", "rare", "mythic"}
    balanced_df = balanced_df.loc[
        (balanced_df["rarity"].isin(valid_rarities)) &
        (balanced_df["set"].isin(global_set_map.keys()))
    ].copy()

    # normalize the 'rarity' and 'set' columns (to match training normalization)
    balanced_df.loc[:, "rarity"] = balanced_df["rarity"].astype(str).str.lower().str.strip()
    balanced_df.loc[:, "set"] = balanced_df["set"].astype(str).str.lower().str.strip()

    ## BALANCED DATASET TEST START - UNCOMMENT TO RUN ON BALANCED DATASET

    '''
    balanced_ds = MTGDataset(balanced_df, img_dir, transform=transform,
                            rarity_map=train_ds.rarity_map, set_map=global_set_map)
    balanced_loader = DataLoader(balanced_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    balanced_loss, (balanced_acc_rar, balanced_acc_col, balanced_acc_set) = evaluate(model, balanced_loader, criterion, device)
    print("\nBalanced Dataset Evaluation:")
    print(f"  Loss={balanced_loss:.4f}")
    print(f"  Acc: Rarity={balanced_acc_rar*100:.1f}%  Color={balanced_acc_col*100:.1f}%  Set={balanced_acc_set*100:.1f}%")
    '''
    ## BALANCED DATASET TEST END

    ##### SINGLE IMAGE CLASSIFICATION START - UNCOMMENT TO CLASSIFY SINGLE IMAGE

    # path_to_img = os.path.join(img_dir, "Break_Down_the_Door_-_dsk_170.jpg")
    # path_to_img = os.path.join(img_dir, "Return_to_Dust_-_tsr_37.jpg")
    path_to_img = os.path.join(img_dir, "Shark_Typhoon_-_otc_113.jpg")
    
    result = model.classify(path_to_img)
    
    reverse_rarity_map = {v: k for k, v in train_ds.rarity_map.items()}
    reverse_set_map = {v: k for k, v in global_set_map.items()}
    def color_index_to_labels(color_index):
        mapping = {"w": 1, "u": 2, "b": 4, "r": 8, "g": 16}
        labels = [c for c, bit in mapping.items() if color_index & bit]
        if not labels:
            labels.append("colorless")
        return labels
    result_labels = {
        "rarity": reverse_rarity_map[result["rarity"]],
        "color": color_index_to_labels(result["color"]),
        "set": reverse_set_map[result["set"]]
    }
    print("Single image classification (labels):", result_labels)

    ##### SINGLE IMAGE CLASSIFICATION END


def get_ground_truth_labels(image_path, csv_path, img_dir):
    # load data and normalize (set to all lowercase, no spaces)
    df = pd.read_csv(csv_path, low_memory=False)
    df.loc[:, "rarity"] = df["rarity"].astype(str).str.lower().str.strip()
    df.loc[:, "set"] = df["set"].astype(str).str.lower().str.strip()
    valid_rarities = {"common", "uncommon", "rare", "mythic"}
    df = df[df["rarity"].isin(valid_rarities)]
    df = df[df["set"] != ""].copy()
    # create a new column with the base name of the image file
    df["img_base"] = df["file_name"].apply(lambda x: os.path.basename(x))
    base_img = os.path.basename(image_path)
    gt_rows = df[df["img_base"] == base_img]
    if gt_rows.empty:
        return None
    row = gt_rows.iloc[0]
    gt_rarity = row["rarity"]
    try:
        colors_list = ast.literal_eval(row["colors"])
    except Exception:
        colors_list = []
    # use the same bitwise encoding
    gt_color_index = MTGDataset.colors_to_index(colors_list)
    # convert color bitmask back to labels.
    def color_index_to_labels(color_index):
        mapping = {"w": 1, "u": 2, "b": 4, "r": 8, "g": 16}
        labels = [c for c, bit in mapping.items() if color_index & bit]
        if not labels:
            labels.append("colorless")
        return labels
    gt_colors = color_index_to_labels(gt_color_index)
    gt_set = row["set"]
    return {"rarity": gt_rarity, "color": gt_colors, "set": gt_set}



if __name__ == "__main__":
    code_dir = os.path.dirname(__file__)
    img_dir = os.path.join("data", "images", "all")
    # Other test images (uncomment to use):
    # img_name = "Liliana,_the_Last_Hope_-_2x2_333.jpg"
    # img_name = "The_Flame_of_Keld_-_dom_123.jpg"
    # img_name = "Bind____Liberate_-_cmb1_88.jpg"
    # img_name = "B.F.M._(Big_Furry_Monster)_-_ugl_29.jpg"
    img_name = "B.F.M._(Big_Furry_Monster)_-_ugl_28.jpg"
    
    image_file = os.path.join(img_dir, img_name)

    df = pd.read_csv('code/master_data.csv', low_memory=False)
    truth = df.loc[df['file_name'] == image_file][['set', 'colors', 'rarity']].iloc[0]

    results = classify_image(image_file)
    print("Classifying image: ", img_name)
    print("real:     ", truth.to_dict())
    print("predicted:", results)





