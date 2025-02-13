import numpy as np
import streamlit as st
from PIL import Image
import os
import cv2
import pickle
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity

# Initialize face detector and VGGFace model
detector = MTCNN()
model = VGGFace(model="resnet50", include_top=False, input_shape=(224, 224, 3), pooling="avg")

# Load precomputed feature list and filenames
feature_list = pickle.load(open("embedding.pkl", "rb"))
filenames = pickle.load(open("filenames.pkl", "rb"))

# Function to save uploaded image
def save_uploaded_image(uploaded_image):
    try:
        upload_path = os.path.join("uploads", uploaded_image.name)
        with open(upload_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        return upload_path
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return None

# Function to extract features from an image
def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    if img is None:
        st.error("Error: Could not load the image. Check the file path.")
        return None

    faces = detector.detect_faces(img)
    if len(faces) > 0:
        # Get face coordinates
        x, y, width, height = faces[0]['box']
        
        # Ensure bounding box is within image bounds
        h, w, _ = img.shape
        x = max(0, x)
        y = max(0, y)
        width = min(w - x, width)
        height = min(h - y, height)

        # Crop and preprocess face
        face = img[y:y+height, x:x+width]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        face = cv2.resize(face, (224, 224))

        # Convert image to float32 and normalize
        face = np.asarray(face, dtype=np.float32)
        face = np.expand_dims(face, axis=0)
        preprocessed_face = preprocess_input(face)  # Normalize image

        # Extract features using VGGFace
        features = model.predict(preprocessed_face).flatten()
        return features
    else:
        return None

# Function to recommend the most similar celebrity
def recommend(feature_list, feature):
    similarity = cosine_similarity(feature.reshape(1, -1), feature_list)
    index_pos = np.argmax(similarity)  # Get highest similarity index
    return index_pos

# Streamlit UI
st.title("Which Bollywood Celebrity Are You?")

# Upload an image
uploaded_image = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_image is not None:
    # Display the uploaded image
    display_image = Image.open(uploaded_image)
    st.image(display_image, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image
    uploaded_img_path = save_uploaded_image(uploaded_image)
    if uploaded_img_path:
        # Extract features from the image
        features = extract_features(uploaded_img_path, model, detector)

        if features is not None:
            st.text(f"Extracted Features Shape: {features.shape}")

            # Recommend the most similar celebrity
            index_pos = recommend(feature_list, features)

            # Extract the correct celebrity name
            celebrity_folder_name = os.path.basename(os.path.dirname(filenames[index_pos]))  # Extract folder name
            celebrity_name = celebrity_folder_name.replace("_", " ")  # Convert underscores to spaces

            st.subheader(f"Matched Celebrity: {celebrity_name}")  # Display the celebrity name

            # Display images side by side
            col1, col2 = st.columns(2)
            with col1:
                st.header("Your Uploaded Image")
                st.image(display_image, caption="Your Image", use_column_width=True)

            with col2:
                st.header(celebrity_name)
                celebrity_image = Image.open(filenames[index_pos])  # Open the celebrity image
                st.image(celebrity_image, caption=f"{celebrity_name}", use_column_width=True)
        else:
            st.error("No face detected in the uploaded image.")



