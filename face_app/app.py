from flask import Flask, request, render_template
import os
import cv2
import face_recognition
import pickle
import numpy as np

app = Flask(__name__)

# Load known face encodings and names from the pickle file
with open(r"D:\Face Detection\face_data.pkl", 'rb') as f:  # Adjust path if necessary
    known_face_encodings, known_face_names = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')  # Render the upload form

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    # Save the uploaded file
    image_path = os.path.join('uploads', file.filename)
    file.save(image_path)

    # Call your prediction function
    processed_image, predicted_names = predict_multiple_faces(image_path, known_face_encodings, known_face_names)

    # Save the processed image for display
    result_image_path = os.path.join('static', 'result.jpg')
    cv2.imwrite(result_image_path, processed_image)

    return render_template('result.html', predicted_names=predicted_names, result_image=result_image_path)

def predict_multiple_faces(image_path, known_face_encodings, known_face_names, threshold=0.4):
    # Load the image
    img = face_recognition.load_image_file(image_path)
    
    # Find face locations and encodings in the image
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    # Convert image to BGR format for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Initialize a list to store predicted names
    predicted_names = []

    # Loop through detected faces and match them with known faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compute distances between the face encoding and known faces
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        # Determine if the face is identified or unidentified
        if face_distances[best_match_index] < threshold:
            name = known_face_names[best_match_index]
            color = (0, 255, 0)  # Green for identified
        else:
            name = "unidentified"
            color = (0, 0, 255)  # Red for unidentified
        
        # Draw rectangle around the face and put the label
        cv2.rectangle(img_bgr, (left, top), (right, bottom), color, 2)
        cv2.putText(img_bgr, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Add the predicted name to the list
        predicted_names.append(name)

    # Return the processed image and predicted names
    return img_bgr, predicted_names

if __name__ == '__main__':
    app.run(debug=True)
