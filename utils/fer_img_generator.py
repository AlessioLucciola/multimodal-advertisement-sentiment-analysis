import os
import cv2
import numpy as np
import pandas as pd
from config import DATASET_NAME, DATASET_DIR
from shared.constants import FER_emotion_mapping

# Load the FER dataset CSV file
csv_file = os.path.join(DATASET_DIR, DATASET_NAME + ".csv")
data = pd.read_csv(csv_file)
emotion_mapping = FER_emotion_mapping

def generate_fer_img():
    # Iterate through the dataset and generate images
    for index, row in data.iterrows():
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


if __name__ == "__main__":
    generate_fer_img()