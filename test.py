from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
from PIL import Image
import pickle

# Load precomputed features and filenames
feature_list = np.array(pickle.load(open("embedding.pkl", "rb")))
filenames = pickle.load(open("filenames.pkl", "rb"))

# Load VGGFace model
model = VGGFace(model="resnet50", include_top=False, input_shape=(224, 224, 3), pooling="avg")

# Initialize face detector
detector = MTCNN()

# Read the sample image
sample_img = cv2.imread("sample/my_photo.jpg")
if sample_img is None:
    print("Error: Could not load the image. Check the file path.")
    exit()

# Detect faces in the image
results = detector.detect_faces(sample_img)
if results:
    x, y, width, height = results[0]["box"]
    h, w, _ = sample_img.shape

    # Clamp coordinates to valid image bounds
    x = max(0, x)
    y = max(0, y)
    width = min(w - x, width)
    height = min(h - y, height)

    face = sample_img[y:y + height, x:x + width]

    # Preprocess the face
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image, dtype=np.float32)
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    
    # Extract features using VGGFace
    result = model.predict(preprocessed_img).flatten()
    print("Extracted Features Shape:", result.shape)

    # Finding the cosine distance of the current image with all the 8655 features
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(result.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    # Sort the similarity scores in descending order
    index_pos = sorted(list(enumerate(similarity)), key=lambda x: x[1], reverse=True)

    # Get the most similar image's filename
    most_similar_index = index_pos[0][0]
    most_similar_filename = filenames[most_similar_index]

    # Load and display the most similar image
    temp_img = cv2.imread(most_similar_filename)
    if temp_img is not None:
        cv2.imshow("Most Similar Image", temp_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Could not load the most similar image.")
else:
    print("Error: No faces detected in the sample image.")
