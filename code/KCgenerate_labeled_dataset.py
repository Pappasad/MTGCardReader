import os
import pandas as pd
import random
import shutil

def generate_labeled_dataset(csv_filename="yolo.csv", train_ratio=0.8):
    """
    Reads the CSV file from data/images, assumes it contains annotations for a small
    set (e.g., 50 images) that you manually labeled via Roboflow. Then randomly splits the unique 
    filenames into training (80%) and validation (20%) sets, copies the images from 
    data/images/all to data/images/labeled_train and data/images/labeled_val, and creates YOLO-format
    annotation text files for each image.
    """
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "images")
    csv_path = os.path.join(base_dir, csv_filename)
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Get unique filenames from the CSV.
    filenames = df["filename"].unique().tolist()
    random.shuffle(filenames)
    split_idx = int(len(filenames) * train_ratio)
    train_files = filenames[:split_idx]
    val_files = filenames[split_idx:]
    
    # Directories for labeled images.
    train_dir = os.path.join(base_dir, "labeled_train")
    val_dir = os.path.join(base_dir, "labeled_val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Mapping from class name to class ID.
    mapping = {"artwork": 0, "text_box": 1, "mana_symbol": 2, "set_symbol": 3, "power_toughness": 4}
    
    def create_annotation_file(filename, dest_dir):
        sub_df = df[df["filename"] == filename]
        base_name = os.path.splitext(filename)[0]
        txt_filename = base_name + ".txt"
        txt_path = os.path.join(dest_dir, txt_filename)
        with open(txt_path, "w") as f:
            for _, row in sub_df.iterrows():
                class_id = mapping.get(row["class"], -1)
                if class_id == -1:
                    continue
                line = f"{class_id} {row['x_center']} {row['y_center']} {row['box_width']} {row['box_height']}\n"
                f.write(line)
    
    all_dir = os.path.join(base_dir, "all")
    for filename in train_files:
        src = os.path.join(all_dir, filename)
        dst = os.path.join(train_dir, filename)
        if os.path.exists(src):
            shutil.copy(src, dst)
            create_annotation_file(filename, train_dir)
        else:
            print(f"Warning: {src} not found.")
    
    for filename in val_files:
        src = os.path.join(all_dir, filename)
        dst = os.path.join(val_dir, filename)
        if os.path.exists(src):
            shutil.copy(src, dst)
            create_annotation_file(filename, val_dir)
        else:
            print(f"Warning: {src} not found.")
    
    print(f"Generated labeled dataset: {len(train_files)} images in labeled_train, {len(val_files)} in labeled_val.")

if __name__ == "__main__":
    generate_labeled_dataset()
