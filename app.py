from flask import Flask, request, jsonify
from deepface import DeepFace
import numpy as np
import tempfile
import os
from PIL import Image

app = Flask(__name__)

# Define the threshold for similarity (70%)
THRESHOLD = 0.7

# Helper function to save numpy array images temporarily and return file paths
def save_temp_image(image_array):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.close()
        Image.fromarray(image_array).save(temp_file.name)
        return temp_file.name
    except Exception as e:
        print(f"Error saving temporary image: {e}")
        raise

@app.route('/compare', methods=['POST'])
def compare_faces():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Both image1 and image2 are required'}), 400

    try:
        # Get the uploaded images
        image1 = request.files['image1']
        image2 = request.files['image2']

        # Convert the images to numpy arrays
        image1_array = np.array(Image.open(image1).convert("RGB"))
        image2_array = np.array(Image.open(image2).convert("RGB"))

        # Save the images as temporary files
        temp_image1_path = save_temp_image(image1_array)
        temp_image2_path = save_temp_image(image2_array)

        # Print debug information about the image paths
        print(f"Temporary Image 1 Path: {temp_image1_path}")
        print(f"Temporary Image 2 Path: {temp_image2_path}")
        print(f"Image 1 Exists: {os.path.exists(temp_image1_path)}")
        print(f"Image 2 Exists: {os.path.exists(temp_image2_path)}")

        # Verify images using DeepFace
        result = DeepFace.verify(
            img1_path=temp_image1_path,
            img2_path=temp_image2_path,
            model_name='ArcFace',
            detector_backend='retinaface'
        )

        # Calculate the similarity based on the distance
        similarity = 1 - result['distance']  # DeepFace returns a distance measure

        # Determine if the faces match based on the threshold
        is_match = similarity >= THRESHOLD

        # Remove temporary files
        os.remove(temp_image1_path)
        os.remove(temp_image2_path)

        return jsonify({
            'cosine_similarity': similarity,
            'result': is_match
        }), 200

    except Exception as e:
        # Print error details
        print(f"Error processing images: {str(e)}")
        return jsonify({'error': f'Exception while processing images: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
