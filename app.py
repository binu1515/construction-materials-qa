import os
from flask import Flask, render_template, request, jsonify, session
import cv2
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from collections import Counter
from datetime import datetime
from huggingface_hub import InferenceClient

# Set Hugging Face API key for authentication
os.environ["HUGGINGFACE_TOKEN"] = "hugging face token"  # Replace with your actual Hugging Face token

# Initialize Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'your-secret-key-here'

# Initialize YOLO model for custom object detection
yolo_model = YOLO('bestx.pt')  # Path to custom YOLO model
yolo_model.to('cuda')
# Initialize InferenceClient for Hugging Face API interaction
client = InferenceClient(api_key=os.getenv("HUGGINGFACE_TOKEN"))

def query_phi_model(messages):
    """Query the Hugging Face Phi model using InferenceClient."""
    try:
        completion = client.chat.completions.create(
            model="microsoft/Phi-3.5-mini-instruct",  # Replace with the correct model name
            messages=messages,
            max_tokens=500
        )
        print(completion) #Log the complete response
        return completion.choices[0].message['content']  # Get the content of the response
    except Exception as e:
        return f"Error during inference: {str(e)}"


#test_message = [{"role": "user", "content": "What objects were detected in the video?"}]
#print(query_phi_model(test_message))

# Store detection results in session
detection_context = {}

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}

# Function to process the video and detect objects using YOLO
def process_video(video_path):
    """Process video and return path to processed video and detection summary."""
    cap = cv2.VideoCapture(video_path)

    # Define class names explicitly (ensure they match your trained YOLO model)
    class_names = ['rod', 'cementbag', 'sand', 'msand', 'sheet', 'redbrick']  # Replace with your actual class names

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_video.mp4')
    if not os.path.exists(output_path):
        print(f"Error: File {output_path} not found!")  # Debugging
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    all_detections = []
    timestamp_detections = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_count / fps
        results = yolo_model(frame)  # Run YOLO model for object detection

        frame_detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = int(box.cls[0])  # Class index
                name = class_names[cls]  # Get class name from the predefined list

                all_detections.append(name)
                frame_detections.append({
                    'object': name,
                    'confidence': float(conf),
                    'timestamp': timestamp
                })

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{name} {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if frame_detections:
            timestamp_detections.append({
                'timestamp': timestamp,
                'detections': frame_detections
            })

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    summary = dict(Counter(all_detections))

    return output_path.replace('static/', '/static/'), summary, timestamp_detections

# Function to generate a response using the Phi model based on video analysis context
def generate_chat_response(question, context):
    """Generate response using Phi model based on the detection context."""
    try:
        # Prepare the context and question
        prompt = f"""
        Video Analysis Context:
        Detected objects: {context['summary']}
        Timeline: {context['timeline']}
        
        Question: {question}
        
        Detailed answer based on the video analysis:
        """

        # Prepare messages for the API
        messages = [{"role": "user", "content": prompt}]
        
        # Get the response from the Hugging Face Phi model
        response = query_phi_model(messages)
        return response

    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the video, perform object detection with YOLO, and generate summary
        processed_video_path, summary, timeline = process_video(filepath)
        
        # Save detection results in the session
        session_id = datetime.now().strftime('%Y%m%d%H%M%S')
        detection_context[session_id] = {
            'summary': summary,
            'timeline': timeline
        }
        
        # Return the processed video and summary
        relative_video_path = os.path.join('uploads', 'processed_video.mp4').replace('\\', '/')
        #relative_video_path = processed_video_path.replace('static/', '')
        print(f"Processed video path: {relative_video_path}")  # Debugging
        return jsonify({
            'message': 'Video processed successfully',
            'video_path': f"/static/{relative_video_path}",
            'summary': summary,
            'session_id': session_id
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    session_id = data.get('session_id')
    
    if not question or not session_id:
        return jsonify({'error': 'Missing question or session ID'}), 400
    
    if session_id not in detection_context:
        return jsonify({'error': 'Invalid session ID'}), 400

    # Get detection summary from session
    detection_summary = detection_context[session_id]['summary']  # Example: {'rod': 5, 'cementbag': 3}
    
    # Format the input for the LLM
    formatted_input = {
        "role": "user",
        "content": f"The detected objects in the video are: {detection_summary}. {question}"
    }
    
    # Send the formatted input to the LLM
    response = query_phi_model([formatted_input])
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
