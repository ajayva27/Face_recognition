# Face Detection & Recognition Application

A face detection and recognition application built using Python, OpenCV, and machine learning models. This project allows users to upload images, detect faces, and identify recognized individuals. It uses a pre-trained model for face detection and a custom dataset for face recognition.

## Features

- Detect faces in images using OpenCV.
- Recognize faces from a custom dataset.
- Identify and differentiate between known and unknown faces.
- Display prediction results visually in images.
- Upload functionality (to be implemented via Flask for web app).

## Installation

Follow these steps to set up the project:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/face-detection.git
cd face-detection

2. Create Virtual Environment

  python -m venv face_env

Activate the virtual environment:

Windows:
face_env\Scripts\activate

macOS/Linux:
source face_env/bin/activate


3. Install Dependencies
pip install -r requirements.txt

4. Run the Jupyter Notebook

To run the Jupyter Notebook for testing the project, use:
jupyter notebook Face_detection.ipynb


5. Flask Web Application
cd face_app
python app.py
