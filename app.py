from flask import Flask, render_template, request
import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import base64

app = Flask(__name__)

# Ensure the directories exist
genuine_images_path = "genuine"
forged_images_path = "forged"

if not os.path.exists(genuine_images_path) or not os.path.exists(forged_images_path):
    raise FileNotFoundError("Ensure 'genuine' and 'forged' directories exist with images.")

# List images in directories
genuine_image_filenames = os.listdir(genuine_images_path)
forged_image_filenames = os.listdir(forged_images_path)

genuine_image_features = []
forged_image_features = []

# Function to preprocess an image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Apply thresholding
    _, threshold_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Ensure the image is not empty before resizing
    if threshold_image is None or threshold_image.size == 0:
        raise ValueError(f"Thresholding failed for image: {image_path}")

    resized_image = cv2.resize(threshold_image, (200, 100))
    return resized_image.flatten()

# Extract features from images
for name in genuine_image_filenames:
    image_path = os.path.join(genuine_images_path, name)
    try:
        feature_vector = preprocess_image(image_path)
        genuine_image_features.append(feature_vector)
    except ValueError as e:
        print(e)

for name in forged_image_filenames:
    image_path = os.path.join(forged_images_path, name)
    try:
        feature_vector = preprocess_image(image_path)
        forged_image_features.append(feature_vector)
    except ValueError as e:
        print(e)

# Ensure we have enough images
if not genuine_image_features or not forged_image_features:
    raise ValueError("No valid images found in 'genuine' or 'forged' folders.")

# Labels (1 for genuine, 0 for forged)
genuine_labels = np.ones(len(genuine_image_features))
forged_labels = np.zeros(len(forged_image_features))

# Combine data
all_features = genuine_image_features + forged_image_features
all_labels = np.concatenate((genuine_labels, forged_labels))

# Convert to NumPy arrays
all_features = np.array(all_features)

# Normalize feature vectors
scaler = StandardScaler()
scaled_features = scaler.fit_transform(all_features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, all_labels, test_size=0.2, random_state=42)

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Save the trained model and scaler
with open("model.pkl", "wb") as file:
    pickle.dump(svm_model, file)
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# Load model and scaler before defining routes
with open("model.pkl", "rb") as file:
    svm_model = pickle.load(file)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    image_file = request.files['image']
    
    if not image_file:
        return "No image uploaded", 400

    image_path = 'temp.jpg'
    image_file.save(image_path)

    try:
        feature_vector = preprocess_image(image_path)
        scaled_feature = scaler.transform(feature_vector.reshape(1, -1))
        prediction = svm_model.predict(scaled_feature)

        # Convert prediction to readable result
        result = "The signature is genuine." if prediction == 1 else "The signature is forged."

        # Convert preprocessed image to base64
        reshaped_image = feature_vector.reshape(100, 200)
        _, encoded_image = cv2.imencode('.png', reshaped_image)
        preprocessed_image_base64 = base64.b64encode(encoded_image).decode('utf-8')

    except ValueError as e:
        result = str(e)
        preprocessed_image_base64 = ""

    # Clean up the temporary file
    os.remove(image_path)

    return render_template('result.html', result=result, processed_image=preprocessed_image_base64)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
