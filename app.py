from flask import Flask, request, jsonify
from deepface import DeepFace
import io
from PIL import Image
import numpy as np

app = Flask(__name__)

# Define the threshold for similarity (60%)
THRESHOLD = 0.6

# Helper function to convert an uploaded image to a numpy array
def convert_image_to_array(image_file):
    # Open the image file
    image = Image.open(image_file)
    # Convert the image to RGB if it's not in that format
    image = image.convert("RGB")
    # Convert the image to a numpy array
    image_np = np.array(image)
    return image_np

@app.route('/compare', methods=['POST'])
def compare_faces():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Both image1 and image2 are required'}), 400

    # Get the uploaded images
    image1 = request.files['image1']
    image2 = request.files['image2']

    # Convert the images to numpy arrays
    image1_array = convert_image_to_array(image1)
    image2_array = convert_image_to_array(image2)

    try:
        # Use DeepFace to analyze the similarity using Facenet512
        result = DeepFace.verify(image1_array, image2_array, model_name='Facenet512')
        
        # Calculate the similarity based on the distance
        similarity = 1 - result['distance']  # DeepFace returns a distance measure

        # Determine if the faces match based on the threshold
        is_match = similarity >= THRESHOLD

        return jsonify({
            'similarity': similarity,
            'match': is_match
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
