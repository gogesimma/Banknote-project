import numpy as np
import cv2
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import tkinter as tk
from tkinter import filedialog
import joblib

# Load the saved VGG model
loaded_VGG_model = load_model('vgg_model.h5')

# Load the pre-trained MLP classifier model
mlp_classifier = joblib.load('mlp_model.pkl')

# Function to preprocess the image
def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Resize the image to match the input shape expected by VGG
    resized_image = cv2.resize(image, (256, 256))
    # Convert image to array and preprocess the image using the preprocess_input function from VGG
    preprocessed_image = preprocess_input(resized_image)
    return preprocessed_image

# Function to extract features using the loaded VGG model
def extract_features(image):
    # Expand the dimensions to create a batch of one image
    preprocessed_image = np.expand_dims(image, axis=0)
    # Extract features using the VGG model
    features = loaded_VGG_model.predict(preprocessed_image)
    # Flatten the features to match the input shape of the MLP classifier
    flattened_features = features.flatten().reshape(1, -1)
    return flattened_features

# Function to handle image upload and feature extraction
def handle_image_upload():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        # Preprocess the uploaded image
        preprocessed_image = preprocess_image(file_path)
        # Extract features from the image using the loaded VGG model
        image_features = extract_features(preprocessed_image)
        # Make a prediction using the MLP classifier
        prediction = mlp_classifier.predict(image_features)
        if prediction[0]<=3:
            print("R100")
        elif prediction[0]>=4 and  prediction[0]<=7:
            print("R10")
        elif prediction[0]>=7 and  prediction[0]<=11:
            print("R200")
        elif prediction[0]>=11 and  prediction[0]<=15:
            print("R20")
        elif prediction[0]>=15 and  prediction[0]<=19:
             print("R50")
# Create the main application window
root = tk.Tk()
root.title("Feature Extractor and Classifier")

# Create a button to upload an image
upload_button = tk.Button(root, text="Upload Image", command=handle_image_upload)
upload_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
