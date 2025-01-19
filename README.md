Material Detection App with Chat

Overview

This project is a web-based application for detecting materials in videos using a YOLO-based custom object detection model. The app processes video uploads, performs object detection, and allows users to interact with a Large Language Model (LLM) to ask questions about the detection results. The processed video and detection summary are presented in an intuitive user interface.
![Screenshot 2025-01-18 155003](https://github.com/user-attachments/assets/3d917f39-3c7c-47f9-8a30-a527209f6e3b)

Features

Video Upload and Processing:

Users can upload videos in formats such as .mp4, .avi, and .mov.

The YOLO model processes the video to detect objects and generates bounding boxes.

Object Detection:

Custom YOLO model (bestx.pt) is used for detecting predefined objects such as rod, cementbag, sand, msand, sheet, and redbrick.
(currently it is not added to the repository)

The training code for building the yolo model is added as a colab notebook

Interactive Chat Interface:

Users can ask questions about the detection results.

The app integrates a Hugging Face LLM (Phi-3.5-mini-instruct) to provide answers based on the detected objects and their counts.

Processed Video Playback:

The processed video with bounding boxes is displayed in the UI for review.

Detection Summary:

A detailed summary of detected objects and their counts is displayed.

Tech Stack

Frontend: HTML, Tailwind CSS, JavaScript

Backend: Flask

Object Detection Model: YOLO (Ultralytics)

LLM Integration: Hugging Face Inference Client

Video Processing: OpenCV

Setup and Installation

Prerequisites

Python 3.8+

Virtual environment tool (optional but recommended)

Installation Steps

Clone the repository:

git clone https://github.com/binu1515/construction-materials-qa.git
cd construction-materials-qa

Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Add your Hugging Face API key:

Replace your-hugging-face-token in the code with your actual API token.

os.environ["HUGGINGFACE_TOKEN"] = "your-hugging-face-token"

Place the YOLO model file (bestx.pt) in the project directory.

Run the application:

python app.py

Open your browser and navigate to:

http://127.0.0.1:5000/

Project Structure

material-detection-app/
|-- static/
|   |-- uploads/               # Stores uploaded and processed videos
|-- templates/
|   |-- index.html             # Frontend UI
|-- app.py                     # Main Flask application
|-- requirements.txt           # Python dependencies
|-- bestx.pt                   # YOLO model file

Usage

Open the application in your browser.

Upload a video file in the supported format.

Wait for the video processing to complete.

View the processed video with bounding boxes and the detection summary.

Use the chat interface to ask questions about the detection results.

Example Questions for Chat

"How many rods were detected?"

"Which object was detected the most?"

"What are the total counts of all objects?"



YOLO Model: 

Detected Classes: Update the class_names list in app.py to match your model's classes.

LLM Model: Replace microsoft/Phi-3.5-mini-instruct with another Hugging Face-supported LLM if needed.

Further enhancements for Performance Optimization

Async Processing: Implement background job processing (e.g., Celery, RQ) for handling 

Video Chunking: Process videos frame-by-frame in smaller chunks to reduce memory usage.

Caching: Cache detection results for faster retrieval in subsequent requests.

Future Improvements

Add real-time video streaming for detection.

Expand the detection classes for other materials.

Integrate GPU acceleration for faster processing.

Enhance the chat interface for more natural conversations.

License

This project is licensed under the MIT License

Acknowledgements

Ultralytics YOLO

Hugging Face Inference API

Flask Framework

OpenCV

