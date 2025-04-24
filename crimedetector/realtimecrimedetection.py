import os
import cv2  # OpenCV for video processing
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
from emailConfiguration import send_alert

# Paths to your models
current_dir = os.path.dirname(os.path.abspath(__file__))
crime_model_path = os.path.join(current_dir, "crime_detector")
crime_type_model_path = os.path.join(current_dir, "crime_type_classifier")


crime_model = AutoModelForImageClassification.from_pretrained("NyanjaCyane/crime-detector",ignore_mismatched_sizes=True)
crime_processor = AutoImageProcessor.from_pretrained("NyanjaCyane/crime-detector")

#crime_model.classifier = torch.nn.Linear(in_features=768, out_features=2)

crime_type_model = AutoModelForImageClassification.from_pretrained("NyanjaCyane/crimes-classifier",ignore_mismatched_sizes=True)
crime_type_processor = AutoImageProcessor.from_pretrained("NyanjaCyane/crimes-classifier")
#crime_type_model.classifier = torch.nn.Linear(in_features=768, out_features=3)

# Open the video file or use the camera (replace 'video.mp4' with 0 for webcam)
video_path = os.path.join(current_dir, "test_video.mp4")
cap = cv2.VideoCapture(video_path)

# Frame processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame (BGR to RGB)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    # Preprocess the frame for the crime detection model
    inputs = crime_processor(images=pil_image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = crime_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        crime_score = probs[0].max().item()
        predicted_label = crime_model.config.id2label.get(probs[0].argmax().item(), "Unknown")

    # Display the crime detection result
    if crime_score > 0.5 and predicted_label == "Crime":
        print(f"Potential Crime Detected! Score: {round(crime_score, 3)}")
        # Preprocess the frame for the crime type detection model
        inputs = crime_type_processor(images=pil_image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = crime_type_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            top5_probs, top5_indices = torch.topk(probs[0], 5)
        label = crime_type_model.config.id2label.get(top5_indices[0].item(), "Unknown")
        score = round(top5_probs[0].item(), 3)
        send_alert("Camera One",label, score)
        print("Top 5 crime types:")
        for i in range(5):
            label = crime_type_model.config.id2label.get(top5_indices[i].item(), "Unknown")
            score = round(top5_probs[i].item(), 3)
            print(f"{label}: {score}")
    # Display the frame with results (optional)
    cv2.imshow('Video Feed', frame)

    # Press 'q' to exit the video early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
