import os  # Module for interacting with the operating system
import time  # Module for adding delays
import requests  # Library for making HTTP requests
import json  # Library for handling JSON data

# Ensure the 'data' and 'data/images' directories exist
os.makedirs('data', exist_ok=True)
os.makedirs(os.path.join('data', 'images'), exist_ok=True)

# Define paths for storing image data and metadata
ALL_DIR = os.path.join('data', 'images', 'all')
TRAIN_PATH = os.path.join('data', 'images', 'train.json')
VALID_PATH = os.path.join('data', 'images', 'valid.json')
TEST_PATH = os.path.join('data', 'images', 'test.json')

# Ensure the 'data/images/all' directory exists
os.makedirs(ALL_DIR, exist_ok=True)

# URL for downloading bulk card data
BULK_DATA_URL = "https://api.scryfall.com/bulk-data"
# Path for storing the full card data JSON file
BULK_DATA_PATH = os.path.join('data', 'full_data.json')

# Function to import images (downloads card data and images from Scryfall API)
def importImages(fetch=False):
    # Check if data needs to be fetched or if the local file doesn't exist
    if fetch or not os.path.exists(BULK_DATA_PATH):
        # Fetch the bulk data metadata from Scryfall
        bulk_response = requests.get(BULK_DATA_URL).json()
        # Find the URL for downloading the default card data
        bulk_file_url = next(
            (data["download_uri"] for data in bulk_response["data"] if data["type"] == "default_cards"), None
        )
        print("Downloading card data...")
        # Download the full card data
        full_data = requests.get(bulk_file_url).json()
        full_data = [card for card in full_data if card.get('lang') == 'en']
        # Save the downloaded data to a JSON file
        with open(BULK_DATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(full_data, f)
        print("Downloaded card data to", BULK_DATA_PATH)
    else:
        # Load the existing card data from the local file
        with open(BULK_DATA_PATH, 'r', encoding='utf-8') as f:
            full_data = json.load(f)

    print("Downloading images... this could take a sec")
    # Iterate through the card data to download images
    for card_data in full_data:
        if 'image_uris' in card_data:
            # Get the normal resolution image URL
            img_url = card_data['image_uris']['normal']
            # Format the filename using card name, set, and collector number
            name = card_data['name'].replace('/', '_').replace(' ', '_')
            set = card_data['set']
            id = card_data['collector_number']
            img_file = f'{name}_-_{set}_{id}.jpg'
            img_path = os.path.join(ALL_DIR, img_file)

            # Download and save the image if it doesn't already exist locally
            if not os.path.exists(img_path):
                try:
                    img_data = requests.get(img_url)
                    with open(img_path, 'wb') as f:
                        f.write(img_data.content)
                    time.sleep(0.5)  # Add a delay to avoid overloading the server
                except Exception:
                    print(f"<WARNING> Failed to download {name}, {set}")

    # Clean up non-image files from the directory
    for file in os.listdir(ALL_DIR):
        if not file.endswith('.jpg'):
            os.remove((os.path.join(ALL_DIR, file)))

    print("Successfully downloaded all images!")

# Main execution block
if __name__ == '__main__':
    # Download images and card data if the script is run directly
    importImages()
