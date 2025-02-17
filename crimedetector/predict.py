
# from transformers import AutoModelForImageClassification, AutoImageProcessor
# from PIL import Image
# import torch

# # Load the image
# image_path = "./crimedetector/assultImage.jpg"  # Replace with the actual path
# image = Image.open(image_path)

# # Load the first model (Crime Detection) from the local directory
# crime_model = AutoModelForImageClassification.from_pretrained("./crime_detector", local_files_only=True)
# crime_processor = AutoImageProcessor.from_pretrained("./crime_detector")

# # Preprocess the image
# inputs = crime_processor(images=image, return_tensors="pt")

# # Get predictions for crime detection
# with torch.no_grad():
#     outputs = crime_model(**inputs)
#     logits = outputs.logits
#     probs = torch.softmax(logits, dim=1)
#     crime_score = probs[0].max().item()
#     predicted_label = crime_model.config.id2label[probs[0].argmax().item()]

# print(f"Predicted Crime Label: {predicted_label}, Score: {round(crime_score, 3)}")

# # Threshold for detecting a crime
# threshold = 0.5
# if crime_score > threshold and predicted_label == "Crime":
#     print("There is a potential crime. Checking the type...")

#     # Load the second model (Crime Type Detection) from the local directory
#     crime_type_model = AutoModelForImageClassification.from_pretrained("./crime_type_classifier", local_files_only=True)
#     crime_type_processor = AutoImageProcessor.from_pretrained("./crime_type_classifier")

#     # Preprocess the image for the second model
#     inputs = crime_type_processor(images=image, return_tensors="pt")

#     # Get predictions for crime type detection
#     with torch.no_grad():
#         outputs = crime_type_model(**inputs)
#         logits = outputs.logits
#         probs = torch.softmax(logits, dim=1)

#     # Display top crime type predictions
#     print("Top scores for crime types:")
#     for idx, prob in enumerate(probs[0]):
#         print(f"{crime_type_model.config.id2label[idx]}: {round(prob.item(), 3)}")

import os
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch
from emailConfiguration import send_alert;

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths to the model directories
crime_model_path = os.path.join(current_dir, "crime_detector")
crime_type_model_path = os.path.join(current_dir, "crime_type_classifier")

# Load the first model (Crime Detection) with ignore_mismatched_sizes=True
crime_model = AutoModelForImageClassification.from_pretrained(crime_model_path, ignore_mismatched_sizes=True)
crime_processor = AutoImageProcessor.from_pretrained(crime_model_path)

# Replace the classifier with a new one for 2 output classes
crime_model.classifier = torch.nn.Linear(in_features=768, out_features=2)

# Load and preprocess the image
image = Image.open(os.path.join(current_dir, "assultImage.jpg"))
inputs = crime_processor(images=image, return_tensors="pt")

# Make predictions
with torch.no_grad():
    outputs = crime_model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    crime_score = probs[0].max().item()
    predicted_label = crime_model.config.id2label.get(probs[0].argmax().item(), "Unknown")

print(f"Predicted Crime Label: {predicted_label}, Score: {round(crime_score, 3)}")

# If a crime is detected, classify the type
threshold = 0.5
if crime_score > threshold and predicted_label == "Crime":
    print("There is a potential crime. Checking the type...")
    send_alert("Camera One", predicted_label, crime_score)
    # Load the second model (Crime Type Detection)
    crime_type_model = AutoModelForImageClassification.from_pretrained(
        crime_type_model_path, ignore_mismatched_sizes=True
    )
    crime_type_processor = AutoImageProcessor.from_pretrained(crime_type_model_path, use_fast=True)

    # Replace the classifier for 2 output classes
    crime_type_model.classifier = torch.nn.Linear(in_features=768, out_features=3)

    # Preprocess the image for the second model
    inputs = crime_type_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = crime_type_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

    # Display top crime type predictions
    print("Top scores for crime types:")
    for idx, prob in enumerate(probs[0]):
        label = crime_type_model.config.id2label.get(idx, "Unknown")
        print(f"{label}: {round(prob.item(), 3)}")
