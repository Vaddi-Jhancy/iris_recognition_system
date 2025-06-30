import cv2
import numpy as np
import os

# Load dataset
left_images = np.load("results/dataset_left.npy", allow_pickle=True)
right_images = np.load("results/dataset_right.npy", allow_pickle=True)

left_labels = np.load("results/dataset_left_labels.npy", allow_pickle=True)
right_labels = np.load("results/dataset_right_labels.npy", allow_pickle=True)

# Create directories for outputs
os.makedirs("results/localized", exist_ok=True)
os.makedirs("results/normalized", exist_ok=True)

def localize_iris(image, subject_id, eye_side):
    """Detects iris using HoughCircles and saves localized iris images."""
    image = cv2.equalizeHist(image)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                               param1=50, param2=30, minRadius=20, maxRadius=100)

    if circles is not None:
        x, y, r = np.uint16(np.around(circles[0, 0]))
        r = min(r, x, y, image.shape[1] - x, image.shape[0] - y)
        iris = image[y - r:y + r, x - r:x + r]

        if iris.shape[0] > 0 and iris.shape[1] > 0:
            # Save localized image before normalization
            cv2.imwrite(f"results/localized/{subject_id}_{eye_side}.bmp", iris)
            return iris, (x, y, r)
    return None, None

def normalize_iris(image, center, radius, subject_id, eye_side):
    """Applies Daugman’s rubber sheet model and saves normalized images."""
    polar_iris = cv2.warpPolar(image, (64, 64), center, radius, cv2.WARP_POLAR_LINEAR)
    cv2.imwrite(f"results/normalized/{subject_id}_{eye_side}.bmp", polar_iris)
    return polar_iris

# Process images
localized_left, localized_right = [], []
normalized_left, normalized_right = [], []

# Process left eye images
for img, subject_id in zip(left_images, left_labels):
    localized_img, iris_info = localize_iris(img, subject_id, "L")
    if localized_img is not None:
        localized_left.append(localized_img)
        normalized_img = normalize_iris(img, (iris_info[0], iris_info[1]), iris_info[2], subject_id, "L")
        normalized_left.append(normalized_img)

# Process right eye images
for img, subject_id in zip(right_images, right_labels):
    localized_img, iris_info = localize_iris(img, subject_id, "R")
    if localized_img is not None:
        localized_right.append(localized_img)
        normalized_img = normalize_iris(img, (iris_info[0], iris_info[1]), iris_info[2], subject_id, "R")
        normalized_right.append(normalized_img)

# Save lists using `allow_pickle=True`
np.save("results/localized_left.npy", np.array(localized_left, dtype=object), allow_pickle=True)
np.save("results/localized_right.npy", np.array(localized_right, dtype=object), allow_pickle=True)
np.save("results/normalized_left.npy", np.array(normalized_left, dtype=object), allow_pickle=True)
np.save("results/normalized_right.npy", np.array(normalized_right, dtype=object), allow_pickle=True)

print("Localization and normalization completed. Results saved separately.")
