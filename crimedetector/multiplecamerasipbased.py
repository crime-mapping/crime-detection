import os
import cv2
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
from threading import Thread

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to the locally saved models
crime_model_path = os.path.join(current_dir, "crime_detector")
crime_type_model_path = os.path.join(current_dir, "crime_type_classifier")

# Load the crime detection model
crime_model = AutoModelForImageClassification.from_pretrained(crime_model_path, ignore_mismatched_sizes=True)
crime_processor = AutoImageProcessor.from_pretrained(crime_model_path)

# Load the crime type classification model
crime_type_model = AutoModelForImageClassification.from_pretrained(crime_type_model_path, ignore_mismatched_sizes=True)
crime_type_processor = AutoImageProcessor.from_pretrained(crime_type_model_path)

# List of camera sources (local webcam and RTSP streams)
camera_sources = [
    0,  # Local webcam
    "rtsp://192.168.1.100:8080/live",  #
    "rtsp://192.168.1.101:8080/live"  
]

# Function to process a single camera feed
def process_camera(camera_index, source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open stream for camera {camera_index} ({source})")
        return
    
    print(f"Processing camera {camera_index} ({source})...")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 != 0:  # Process every 5th frame for performance optimization
            continue

        # Convert the frame to RGB and preprocess it
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        inputs = crime_processor(images=pil_image, return_tensors="pt")

        # Crime detection prediction
        with torch.no_grad():
            outputs = crime_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            crime_score = probs[0].max().item()
            predicted_label = crime_model.config.id2label.get(probs[0].argmax().item(), "Unknown")

        # If a crime is detected, classify the type
        if crime_score > 0.5 and predicted_label == "Crime":
            print(f"[Camera {camera_index}] Potential Crime Detected! Score: {round(crime_score, 3)}")
            
            # Preprocess and classify the crime type
            inputs = crime_type_processor(images=pil_image, return_tensors="pt")
            with torch.no_grad():
                outputs = crime_type_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                top5_probs, top5_indices = torch.topk(probs[0], 5)

            print(f"[Camera {camera_index}] Top 5 Crime Types:")
            for i in range(5):
                label = crime_type_model.config.id2label.get(top5_indices[i].item(), "Unknown")
                score = round(top5_probs[i].item(), 3)
                print(f"{label}: {score}")

        # Display the video feed with a label
        cv2.putText(frame, f"Camera {camera_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(f"Camera {camera_index}", frame)

        # Press 'q' to stop all streams
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Launch processing for each camera in a separate thread
threads = []
for i, source in enumerate(camera_sources):
    thread = Thread(target=process_camera, args=(i, source))
    thread.start()
    threads.append(thread)

# Wait for all threads to finish
for thread in threads:
    thread.join()
