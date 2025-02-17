
import cv2
import threading
import torch
import requests
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
from emailConfiguration import send_alert;

# Load models
crime_model = AutoModelForImageClassification.from_pretrained("NyanjaCyane/crime-detector",ignore_mismatched_sizes=True)
crime_processor = AutoImageProcessor.from_pretrained("NyanjaCyane/crime-detector")

crime_model.classifier = torch.nn.Linear(in_features=768, out_features=2)

crime_type_model = AutoModelForImageClassification.from_pretrained("NyanjaCyane/crimes-classifier",ignore_mismatched_sizes=True)
crime_type_processor = AutoImageProcessor.from_pretrained("NyanjaCyane/crimes-classifier")
crime_type_model.classifier = torch.nn.Linear(in_features=768, out_features=3)

API_URL = "http://127.0.0.1:8000/api/cameras/"
response = requests.get(API_URL)
backend_camera_sources = [cam["source_url"] for cam in response.json()["cameras"]]
# Camera sources
camera_sources = [
    0,  # Laptop webcam
    1,  # External webcam
    "http://10.10.9.171:4747/video"  # Mobile IP camera
]

# Function to process a camera feed
def process_camera(camera_source, camera_name):
    cap = cv2.VideoCapture(camera_source,cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print(f"❌ Error: Could not open {camera_name}")
        return
    
    print(f"✅ Processing {camera_name}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB and preprocess
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

        # If crime detected, classify the type
        if crime_score > 0.5 and predicted_label == "Crime":
            print(f"[{camera_name}] 🚨 Potential Crime Detected! Score: {round(crime_score, 3)}")
            send_alert(camera_name, predicted_label, crime_score)
            inputs = crime_type_processor(images=pil_image, return_tensors="pt")
            with torch.no_grad():
                outputs = crime_type_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                top5_probs, top5_indices = torch.topk(probs[0], 5)

            print(f"[{camera_name}] Top 5 crime types:")
            for i in range(5):
                label = crime_type_model.config.id2label.get(top5_indices[i].item(), "Unknown")
                score = round(top5_probs[i].item(), 3)
                print(f"{label}: {score}")

        return frame  # Return the frame for OpenCV processing

    cap.release()

# **New Function: Process Video in Main Thread (Fix for OpenCV)**
def display_video():
    caps = [cv2.VideoCapture(src) for src in camera_sources]
    while True:
        frames = [cap.read()[1] for cap in caps]

        for i, frame in enumerate(frames):
            if frame is not None:
                cv2.imshow(f"Camera {i}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

# Start crime detection threads
threads = []
for index, source in enumerate(camera_sources):
    camera_name = f"Camera {index}" if isinstance(source, int) else "Mobile Camera"
    thread = threading.Thread(target=process_camera, args=(source, camera_name))
    thread.start()
    threads.append(thread)

# **Run OpenCV in Main Thread**
display_video()

# Wait for all threads to finish
for thread in threads:
    thread.join()
