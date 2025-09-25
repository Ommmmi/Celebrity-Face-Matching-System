# Celebrity-Face-Matching-System
This project is a Celebrity Face Matching System that utilizes deep learning techniques to identify the closest matching celebrity from a given facial imageIt leverages VGGFace (ResNet50) for facial feature extraction and Cosine Similarity for matching the uploaded image with a precomputed database of celebrity faces.

Features

Deep Learning-based facial recognition using VGGFace.

MTCNN for face detection to extract facial regions from images.

Cosine Similarity for comparison of facial embeddings.

Streamlit-based web interface for easy user interaction.

Fast and scalable implementation for real-time face matching.

Installation

Prerequisite

Ensure you have Python 3.x installed along with the required dependencies.

Clone the Repository

git clone https://github.com/yourusername/Celebrity-Face-Matching.git
cd Celebrity-Face-Matching

Install Dependencies

pip install numpy pandas tensorflow keras opencv-python mtcnn streamlit

Usage

Run the Streamlit application

streamlit run app.py

Upload an image through the web interface.

The system will detect and extract facial features.

It will compute similarity scores and display the most similar celebrity along with their image.

Technologies Used

Python

TensorFlow & Keras (VGGFace)

OpenCV & MTCNN (Face Detection & Image Processing)

Streamlit (Web Interface)

NumPy & Pandas (Data Handling)

Scikit-learn (Cosine Similarity Calculation)

Example Output

Uploaded Image: Your Photo
Matched Celebrity: Shah Rukh Khan
Similarity Score: 98.7%

Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue.

License

This project is licensed under the MIT License.
