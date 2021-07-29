import splitfolders
import shutil

# Defining an input path, which is also an output path
input_path = "data/dataset/"

# Splitting dataset into training set (80%) and test set (20%)
print("[INFO] Splitting into training and test set...")
splitfolders.ratio(input=input_path, output=input_path, seed=42, ratio=(0.8, 0, 0.2))

# Removing undivided data
print("[INFO] Deleting unnecessary folders...")
folders = ['not_smiling', 'smiling']

for folder in folders:
    shutil.rmtree(input_path + folder, ignore_errors=True)

# Removing empty 'val' folder
shutil.rmtree(input_path + "val")
