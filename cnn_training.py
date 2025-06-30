import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Load normalized iris images
X_left = np.load("results1/normalized_left.npy", allow_pickle=True)
X_right = np.load("results1/normalized_right.npy", allow_pickle=True)

# Reshape images for CNN
X_left = np.array([np.array(img).reshape(64, 64, 1) for img in X_left], dtype=np.float32)
X_right = np.array([np.array(img).reshape(64, 64, 1) for img in X_right], dtype=np.float32)

# Labels: Left Eye = 0, Right Eye = 1
y_left = np.zeros(len(X_left), dtype=int)
y_right = np.ones(len(X_right), dtype=int)

# Merge left and right datasets
X = np.vstack((X_left, X_right))
y = np.hstack((y_left, y_right))

# Split into Training (70%), Validation (15%), and Test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

# Define CNN for feature extraction with explicit input layer
def build_feature_extraction_cnn():
    inputs = tf.keras.layers.Input(shape=(64, 64, 1))  # Explicit input layer
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    feature_vector = tf.keras.layers.Dense(128, activation='relu')(x)  # Feature vector layer

    # softmax layer for classification
    output = tf.keras.layers.Dense(2, activation='softmax')(x)  # 2 classes: Left Eye (0), Right Eye (1)

    model = tf.keras.Model(inputs=inputs, outputs=output)  # Use Functional API
    return model

# Train the model (optional if you are training to refine features)
cnn_model = build_feature_extraction_cnn()
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the trained model in Keras format (recommended)
cnn_model.save("results/cnn_feature_extraction_model.keras")

print("Trained CNN model saved!")

# No need to create `feature_extractor`, use `cnn_model` directly!
# Extract features directly using the trained model
X_train_features = cnn_model.predict(X_train)
X_val_features = cnn_model.predict(X_val)
X_test_features = cnn_model.predict(X_test)

# Save the features
np.save("results/X_train_features.npy", X_train_features)
np.save("results/X_val_features.npy", X_val_features)
np.save("results/X_test_features.npy", X_test_features)

print("Feature extraction completed and features saved!")
