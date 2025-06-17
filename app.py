import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# Define function to read and preprocess image
def read_image(image, size):
    image = np.array(image)
    image = cv2.resize(image, (size, size))
    image = image / 255.0
    image = image.astype(np.float32)
    return image

# Load model and labels
@st.cache_resource
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Load label data
@st.cache_data
def load_labels(labels_path):
    labels_df = pd.read_csv(labels_path)
    breed = labels_df["breed"].unique()
    breed2id = {name: i for i, name in enumerate(breed)}
    id2breed = {i: name for i, name in enumerate(breed)}
    return breed2id, id2breed

def main():
    st.title("Dog Breed Classification")
    
    # Sidebar for model and image settings
    st.sidebar.title("Settings")
    size = st.sidebar.slider("Image Size", 128, 512, 224)
    
    # Set paths
    base_path = r"C:\Users\User\Documents\Murdoch University_Sandeep 2023\3rd Semester\Artificial Intelligence\Assignment 2\Dog-Breed-Classification"
    model_path = os.path.join(base_path, "models", "final_model.keras")
    labels_path = os.path.join(base_path, "labels.csv")
    
    # Load model and labels
    model = load_model(model_path)
    breed2id, id2breed = load_labels(labels_path)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        
        # Preprocess the image
        image = read_image(image, size)
        image = np.expand_dims(image, axis=0)
        
        # Make a prediction
        preds = model.predict(image)[0]
        top_3_idx = np.argsort(preds)[-3:][::-1]
        top_3_breeds = [(id2breed[idx], preds[idx]) for idx in top_3_idx]
        
        # Display the result
        st.write("Top-3 Predicted Breeds:")
        for breed_name, confidence in top_3_breeds:
            st.write(f"{breed_name}: {confidence:.4f}")
        
        # Confidence score of the top prediction
        st.write(f"Confidence Score: {top_3_breeds[0][1]:.4f}")
        
        # Plotting the prediction confidence
        fig, ax = plt.subplots()
        breeds = [breed[0] for breed in top_3_breeds]
        confidences = [breed[1] for breed in top_3_breeds]
        ax.barh(breeds, confidences, color='skyblue')
        ax.set_xlabel('Confidence')
        ax.set_title('Top-3 Predicted Breeds')
        ax.invert_yaxis()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
