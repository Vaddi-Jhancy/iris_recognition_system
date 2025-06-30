import os
import cv2
import numpy as np

# Path to the database
database_path = "IITD_database/IITD Database"

# Folders to ignore
ignore_folders = {"001", "002", "003", "004", "005", "006", "007", "008", "009", 
                  "010", "011", "012", "013", "027", "055", "065"}

# Create results directory
os.makedirs("results", exist_ok=True)

# Lists to store images and labels separately
left_images, left_labels = [], []
right_images, right_labels = [], []

# Load images
for subject in sorted(os.listdir(database_path)):
    subject_path = os.path.join(database_path, subject)
    
    if os.path.isdir(subject_path) and subject not in ignore_folders:
        for img_file in sorted(os.listdir(subject_path)):
            if img_file.endswith(".bmp"):
                img_path = os.path.join(subject_path, img_file)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (320, 240))

                if "_L" in img_file:
                    left_images.append(image)
                    left_labels.append(subject)
                elif "_R" in img_file:
                    right_images.append(image)
                    right_labels.append(subject)

print(f"Loaded {len(left_images)} left eye images and {len(right_images)} right eye images.")

# Save images and labels separately
np.save("results/dataset_left.npy", np.array(left_images, dtype=np.uint8))
np.save("results/dataset_left_labels.npy", np.array(left_labels))

np.save("results/dataset_right.npy", np.array(right_images, dtype=np.uint8))
np.save("results/dataset_right_labels.npy", np.array(right_labels))
