from shutil import copyfile
import os
import pandas as pd
import numpy as np


IMAGES_PATH = "data/img_align_celeba/img_align_celeba/"
SMILING_DIR = 'data/dataset/smiling/'
NOT_SMILING_DIR = 'data/dataset/not_smiling/'

# Loading CSV file with attributes
attributes_csv = pd.read_csv('data/list_attr_celeba.csv')

# Extracting image ID and Smiling attribute from CSV file
is_smiling = attributes_csv[['image_id', 'Smiling']]
is_smiling = np.array(is_smiling)


if __name__ == '__main__':
    # Path for images with smiling faces
    if not os.path.exists(SMILING_DIR):
        os.makedirs(SMILING_DIR)

    # Path for images with not smiling faces
    if not os.path.exists(NOT_SMILING_DIR):
        os.makedirs(NOT_SMILING_DIR)

    # Distributing each photo to 'smiling' and 'not_smiling' folders
    for row in is_smiling:
        (image_id, smiling) = row

        if smiling == 1:
            copyfile(IMAGES_PATH + image_id, SMILING_DIR + image_id)
        elif smiling == -1:
            copyfile(IMAGES_PATH + image_id, NOT_SMILING_DIR + image_id)
