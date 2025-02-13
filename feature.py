
import os
import pickle
import numpy as np
from tqdm import tqdm
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from tensorflow.keras.preprocessing import image

# Load the filenames and the VGGFace model
filenames = pickle.load(open("filenames.pkl", "rb"))
model = VGGFace(model="resnet50", include_top=False, input_shape=(224, 224, 3), pooling="avg")

# Function to extract features from an image
def feature_extractor(img_path, model):
    try:
        # Check if the file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File not found: {img_path}")

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img)

        # Extract features from the image using the VGGFace model
        result = model.predict(preprocessed_img).flatten()
        return result

    except FileNotFoundError as fnf_error:
        print(fnf_error)
        return None
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

# Extract features for all images
features = []
for file in tqdm(filenames, desc="Extracting features"):
    feature = feature_extractor(file, model)
    if feature is not None:
        features.append(feature)

# Ensure that features were extracted
if features:
    # Save the sextracted features to a pickle file
    pickle.dump(features, open("embedding.pkl", "wb"))
    print("Feature extraction complete and embeddings saved.")
else:
    print("No features were extracted. Please check your files.")
