import cv2
import numpy as np
from scipy.spatial.distance import cosine, euclidean
import tensorflow as tf

# Load extracted feature vectors (database templates)
features_left = np.load("results/features_left.npy", allow_pickle=True)  # Left-eye features
features_right = np.load("results/features_right.npy", allow_pickle=True)  # Right-eye features
left_labels = np.load("results1/dataset_left_labels.npy", allow_pickle=True)
right_labels = np.load("results1/dataset_right_labels.npy", allow_pickle=True)

# 🔹 Step 1: Load the input query image and preprocess it
def preprocess_query(image_path):
    """Loads and normalizes the input query image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64))  # Resize to CNN input size
    image = image.reshape(1, 64, 64, 1).astype(np.float32) / 255.0  # Normalize and add batch dimension
    return image

# 🔹 Step 2: Extract features from the query image using the trained feature extraction model
def extract_query_features(image_path, model_path="results/cnn_feature_extraction_model.keras"):
    """Extracts deep features from the query image using the feature extraction CNN."""
    feature_model = tf.keras.models.load_model(model_path)
    query_image = preprocess_query(image_path)
    query_features = feature_model.predict(query_image).flatten()  # Flatten the output
    return query_features

# 🔹 Step 3: Compute match scores using different metrics
def compute_match_scores(query_features, database_features):
    """Computes match scores using Cosine Similarity & Euclidean Distance."""
    cosine_scores = np.array([1 - cosine(query_features, db_feature) for db_feature in database_features])
    euclidean_scores = np.array([1 / (1 + euclidean(query_features, db_feature)) for db_feature in database_features])  # Normalize

    return cosine_scores, euclidean_scores

# 🔹 Step 4: Match query image with the database
def match_query_image(query_image_path, eye_side="L"):
    """Matches an input image against the stored database templates."""
    query_features = extract_query_features(query_image_path)

    if eye_side == "L":
        cosine_scores, euclidean_scores = compute_match_scores(query_features, features_left)
        labels = left_labels
    else:
        cosine_scores, euclidean_scores = compute_match_scores(query_features, features_right)
        labels = right_labels

    # Get best match (highest cosine similarity & lowest Euclidean distance)
    best_match_cosine = labels[np.argmax(cosine_scores)]
    best_match_euclidean = labels[np.argmax(euclidean_scores)]

    # Retrieve matched subject ID
    subject_id_cosine = str(best_match_cosine).zfill(3)  # Convert to three-digit format
    subject_id_euclidean = str(best_match_euclidean).zfill(3)

    # Construct matched image filenames
    matched_image_cosine = f"IITD_database/IITD Database/{subject_id_cosine}/01_{eye_side}.bmp"
    matched_image_euclidean = f"IITD_database/IITD Database/{subject_id_euclidean}/01_{eye_side}.bmp"

    print(f"Best match (Cosine Similarity): Subject {best_match_cosine} - Score: {max(cosine_scores):.4f}")
    print(f"Best match (Euclidean Distance): Subject {best_match_euclidean} - Score: {max(euclidean_scores):.4f}")
    print(f"Matched Image (Cosine Similarity): {matched_image_cosine}")
    print(f"Matched Image (Euclidean Distance): {matched_image_euclidean}")

    # Save match scores
    np.savetxt("results/query_match_scores_cosine.csv", cosine_scores, delimiter=",")
    np.savetxt("results/query_match_scores_euclidean.csv", euclidean_scores, delimiter=",")

# 🔹 Example Usage
query_image_path = "IITD_database/IITD Database/014/01_L.bmp"  # Change this path to your query image
match_query_image(query_image_path, eye_side="L")  # Change to "R" for right-eye images
