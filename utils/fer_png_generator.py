import pandas as pd
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from config import *

# Open csv file: data/VIDEO/old/FER/fer2013.csv
# from 'pixels' column, generate images and save them in 'data/VIDEO/old/FER/fer2013_images' folder
# Add 'file_name' column to the csv file and save it in 'data/VIDEO/old/FER/fer2013_metadata.csv'

# Read the csv file
df = pd.read_csv(VIDEO_METADATA_CSV)

# Create the image folder if it does not exist
if not os.path.exists(FRAMES_FILES_DIR):
    os.makedirs(FRAMES_FILES_DIR)

# Generate images from the 'pixels' column and save them in the image folder
for index, row in tqdm(df.iterrows()):
    pixels = row['pixels']
    image_path = os.path.join(FRAMES_FILES_DIR, f'image_{index}.png')
    # Generate the image from the pixels (RGB) and save it
    pixels = pixels.split(' ')
    pixels = [int(pixel) for pixel in pixels]
    pixels = np.array(pixels, dtype='uint8')
    pixels = pixels.reshape((48, 48))
    img = Image.fromarray(pixels)
    img.save(image_path)

# Add 'file_name' column to the csv file
df['frame'] = [f'image_{index}.png' for index in range(len(df))]

# Reorder the columns
df = df[['frame', 'emotion', 'Usage']]

# Save the modified csv file
df.to_csv(VIDEO_METADATA_FRAMES_CSV, index=False)