import os
import cv2
import numpy as np
import pandas as pd
from config import DATASET_NAME, DATASET_DIR
from tqdm import tqdm
from shared.constants import FER_emotion_mapping

# Load the FER dataset CSV file
csv_file = os.path.join(DATASET_DIR, DATASET_NAME + ".csv")
data = pd.read_csv(csv_file)
emotion_mapping = FER_emotion_mapping

def generate_fer_img():
    # Iterate through the dataset and generate images
    for index, row in tqdm(data.iterrows(), total=len(data)):
        pixels = np.array(row["pixels"].split(), dtype=np.uint8)
        image = pixels.reshape((48, 48))
        emotion = row["emotion"]
        type = row["Usage"]

        # Map the emotion label to the corresponding emotion
        emotion = emotion_mapping[emotion]

        # Create a directory for type of image (train, test)
        type_dir = os.path.join(DATASET_DIR, type)
        os.makedirs(type_dir, exist_ok=True)

        # Save the image with the emotion label
        image_path = os.path.join(type_dir, f"{emotion}_{index}.png")
        cv2.imwrite(image_path, image)

def generate_balanced_fer_img():
    # Load the balanced dataset CSV file
    balanced_csv_file = os.path.join(DATASET_DIR, DATASET_NAME + "_balanced.csv")
    balanced_data = pd.read_csv(balanced_csv_file)

    # Iterate through the balanced dataset and generate images
    for index, row in tqdm(balanced_data.iterrows(), total=len(balanced_data)):
        if row["balanced"]:
            pixels = np.array(row["pixels"].split(), dtype=np.uint8)
            image = pixels.reshape((48, 48))
            emotion = row["emotion"]
            type = row["Usage"]

            # Map the emotion label to the corresponding emotion
            emotion = emotion_mapping[emotion]

            # Create a directory for type of image (train, test)
            type_dir = os.path.join(DATASET_DIR, type + "_balanced")
            os.makedirs(type_dir, exist_ok=True)

            # Save the image with the emotion label
            image_path = os.path.join(type_dir, f"{emotion}_{index}.png")
            cv2.imwrite(image_path, image)

def generate_augmented_fer_img():
    # Load the augmented dataset CSV file
    augmented_csv_file = os.path.join(DATASET_DIR, DATASET_NAME + "_augmented.csv")
    augmented_data = pd.read_csv(augmented_csv_file)

    # Iterate through the augmented dataset and generate images
    for index, row in tqdm(augmented_data.iterrows(), total=len(augmented_data)):
        if row["augmented"]:
            pixels = np.array(row["pixels"].split(), dtype=np.uint8)
            image = pixels.reshape((48, 48))
            emotion = row["emotion"]
            type = row["Usage"]

            # Map the emotion label to the corresponding emotion
            emotion = emotion_mapping[emotion]

            # Create a directory for type of image (train, test)
            type_dir = os.path.join(DATASET_DIR, type + "_augmented")
            os.makedirs(type_dir, exist_ok=True)

            # Save the image with the emotion label
            image_path = os.path.join(type_dir, f"{emotion}_{index}.png")
            cv2.imwrite(image_path, image)

if __name__ == "__main__":
    # Choose which function to run
    choice = input("Choose which function to run (1: generate_fer_img, 2: generate_balanced_fer_img, 3: generate_augmented_fer_img): ")

    if choice == "1":
        generate_fer_img()
    elif choice == "2":
        generate_balanced_fer_img()
    elif choice == "3":
        generate_augmented_fer_img()
    else:
        print("Invalid choice. Please choose a valid function to run.")

