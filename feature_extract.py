import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Load normalized iris images
X_left = np.load("results1/normalized_left.npy", allow_pickle=True)[:50]
X_right = np.load("results1/normalized_right.npy", allow_pickle=True)[:50]

# Convert list to numpy array and reshape for CNN input
X_left = np.array([img.reshape(64, 64, 1) for img in X_left], dtype=np.float32) / 255.0
X_right = np.array([img.reshape(64, 64, 1) for img in X_right], dtype=np.float32) / 255.0

# Load trained feature extraction model
full_model = tf.keras.models.load_model("results/cnn_feature_extraction_model.keras")

# Create feature extractor (stops at the last dense layer)
feature_extractor = tf.keras.Model(inputs=full_model.input, outputs=full_model.layers[-1].output)

# Extract features
features_left = feature_extractor.predict(X_left)
features_right = feature_extractor.predict(X_right)

# Save extracted features
np.save("results/features_left.npy", features_left)
np.save("results/features_right.npy", features_right)

# Create directories for saving images
os.makedirs("results/left_images", exist_ok=True)
os.makedirs("results/right_images", exist_ok=True)

# Save left iris images
for i, img in enumerate(X_left):
    plt.imshow(img.reshape(64, 64), cmap='gray')
    plt.axis('off')
    plt.savefig(f"results/left_images/left_{i+1}.png", bbox_inches='tight', pad_inches=0)

# Save right iris images
for i, img in enumerate(X_right):
    plt.imshow(img.reshape(64, 64), cmap='gray')
    plt.axis('off')
    plt.savefig(f"results/right_images/right_{i+1}.png", bbox_inches='tight', pad_inches=0)

print("Feature extraction and image saving completed. Results saved!")
