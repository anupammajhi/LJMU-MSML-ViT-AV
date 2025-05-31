
import os, shutil
import requests
from tqdm import tqdm
import zipfile
import random

START_INDEX = 1
END_INDEX = 7480
NUM_FILES_TO_COPY = 7480

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(dest_path)}",
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def download_and_extract(zip_url, zip_file_path, extract_path):
    if not os.path.exists(extract_path):
        print(f"Dataset not found for {extract_path}, downloading and extracting...")

        download_file(zip_url, zip_file_path)
        os.makedirs(extract_path, exist_ok=True)
        print(f"Extracting {zip_file_path} to {extract_path}...")
        extract_zip(zip_file_path, extract_path)
        os.remove(zip_file_path)

        print(f"Downloaded and extracted to {extract_path}")
    else:
        print(f"Dataset already exists at {extract_path}.")

if __name__ == "__main__":

    # Define paths for the extracted images and labels
    drive_destination_path = '../data'
    images_extracted_path = drive_destination_path + '_object_image_2/training/image_2'
    labels_extracted_path = drive_destination_path + '_object_label_2/training/label_2'

    # Define destination path
    drive_destination_path = '../data/'
    images_path = drive_destination_path + 'images'
    labels_path = drive_destination_path + 'labels'

    # Training Images
    image_zip_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
    image_zip_file = "data_object_image_2.zip"
    image_extract_path = "data_object_image_2"
    download_and_extract(image_zip_url, image_zip_file, image_extract_path)

    # Training Labels
    label_zip_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
    label_zip_file = "data_object_label_2.zip"
    label_extract_path = "data_object_label_2"
    download_and_extract(label_zip_url, label_zip_file, label_extract_path)

    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    # Generate a list of all possible indices
    all_indices = list(range(START_INDEX, END_INDEX + 1))

    # Randomly select a subset of indices
    num_to_select = min(NUM_FILES_TO_COPY, len(all_indices))
    random_indices = random.sample(all_indices, num_to_select)
    print(f"Selecting and copying {num_to_select} random files...")

    for i in random_indices:
        # Format the file name with leading zeros
        file_name = f"{i:06d}.png"
        source_image_file = os.path.join(images_extracted_path, file_name)
        destination_image_file = os.path.join(images_path, file_name)

        if os.path.exists(source_image_file):
            print(f"Working on {source_image_file}")
            shutil.copy(source_image_file, destination_image_file)
        else:
            raise FileNotFoundError(f"Error: Source file not found: {source_image_file}")

        # Copy the corresponding label file
        label_file_name = f"{i:06d}.txt"
        source_label_file = os.path.join(labels_extracted_path, label_file_name)
        destination_label_file = os.path.join(labels_path, label_file_name)

        if os.path.exists(source_label_file):
            print(f"Working on {source_label_file}")
            shutil.copy(source_label_file, destination_label_file)
        else:
            raise FileNotFoundError(f"Error: Source label file not found: {source_label_file}")

    print(f"Copied images and labels from {START_INDEX:06d} to {END_INDEX:06d}")
    print(f"Images copied to: {images_path}")
    print(f"Labels copied to: {labels_path}")

