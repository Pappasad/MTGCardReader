import os
import tkinter as tk
from tkinter import filedialog
from APcardreader import CardReader
from APutil import LABEL_FLAG
from PIL import Image, ImageTk

data = 'images_for_show'

reader = CardReader()

root = tk.Tk()
root.title('CardReader')
root.geometry('1600x800')

text_output = tk.Label(root, text='', font=('Arial, 12'))
text_output.pack(pady=20)

img_label = tk.Label(root)
img_label.pack(side='bottom', pady=10)


def resize_proportional(image, max_width=800, max_height=600):
    # Get original dimensions
    original_width, original_height = image.size  
    # Calculate aspect ratio
    ratio = min(max_width / original_width, max_height / original_height) 
    # New size maintaining aspect ratio
    new_size = (int(original_width * ratio), int(original_height * ratio)) 
    # Resize image
    return image.resize(new_size, Image.Resampling.LANCZOS)

def runModel():
    global img_label, display
    path = filedialog.askopenfilename(initialdir=data)
    if not path:  # If no file is selected, return
        return
    img_label.destroy()
    img = Image.open(path).convert('RGB')
    text_output.config(text='Predicting...')
    resized = resize_proportional(img)
    display = ImageTk.PhotoImage(resized)
    img_label = tk.Label(root, image=display)
    img_label.pack(side='bottom', pady=10)
    root.update()

    name = os.path.basename(path)
    original_name = name[:name.rfind(LABEL_FLAG)].replace('_', ' ')
    original_name = original_name[:original_name.rfind('.')]
    predicted_name, bg = reader(img)
    text_output.config(text=f'Prediction: {predicted_name}\nColor: {bg["color"]}  Rarity: {bg["rarity"]}  Set: {bg["set"]}')

button = tk.Button(root, text='Select File', command=runModel)
button.pack()

print("Created mainloop...")
root.mainloop()