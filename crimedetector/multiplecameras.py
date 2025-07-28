

import os
import cv2
import threading
import torch
import requests
import time
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
from emailConfiguration import send_alert
from image_uploader import upload_image_to_cloudinary
from crime_reporter import CRIME_TYPE_MAP, send_crime_to_api, get_emergency_level

# Load models
crime_model = AutoModelForImageClassification.from_pretrained(
    "NyanjaCyane/crime-detector", ignore_mismatched_sizes=True
)
crime_processor = AutoImageProcessor.from_pretrained("NyanjaCyane/crime-detector")
crime_model.classifier = torch.nn.Linear(in_features=768, out_features=2)

crime_type_model = AutoModelForImageClassification.from_pretrained(
    "NyanjaCyane/crimes-classifier", ignore_mismatched_sizes=True
)
crime_type_processor = AutoImageProcessor.from_pretrained("NyanjaCyane/crimes-classifier")
crime_type_model.classifier = torch.nn.Linear(in_features=768, out_features=13)  # Ensure 13 crimes supported

headers = {
        "model-api-key": os.getenv("MODEL_API_KEY"), 
}
# Fetch camera URLs from API
def fetch_camera_urls():
    try:
        response = requests.get("https://smart-surveillance-system.onrender.com/api/cameras",headers=headers)
        response.raise_for_status()
        cameras = response.json()
        return [
            cam for cam in cameras
            if cam.get("streamUrl") and isinstance(cam.get("location"), dict)
        ]
    except Exception as e:
        print("❌ Failed to fetch camera URLs:", e)
        return []

real_cameras = fetch_camera_urls()

# Process each camera stream
def process_camera(camera):
    camera_source = camera["streamUrl"].strip()
    camera_name = camera.get("name", "Unnamed Camera")
    location_id = camera["location"]["_id"]
    location_name = camera["location"]["location"]

    cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        print(f"❌ Error: Could not open {camera_name}")
        return

    print(f"✅ Processing {camera_name}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Lost frame from {camera_name}")
            break

        try:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            inputs = crime_processor(images=pil_image, return_tensors="pt")

            with torch.no_grad():
                outputs = crime_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                crime_score = probs[0].max().item()
                predicted_label = crime_model.config.id2label.get(probs[0].argmax().item(), "Unknown")

            if crime_score > 0.5 and predicted_label == "Crime":
                print(f"[{camera_name}] 🚨 Detected Crime! Score: {round(crime_score, 3)}")
                inputs = crime_type_processor(images=pil_image, return_tensors="pt")
                with torch.no_grad():
                    outputs = crime_type_model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)
                    k = min(5, probs.shape[1])
                    top_probs, top_indices = torch.topk(probs[0], k)

                top_label_index = top_indices[0].item()
                top_label_score = top_probs[0].item()
                top_label = crime_type_model.config.id2label.get(top_label_index, "Unknown")
                predicted_label_raw = crime_type_model.config.id2label.get(top_label_index, "Unknown")
                crime_type_enum = CRIME_TYPE_MAP.get(predicted_label_raw, "STEALING")  


                send_alert(camera_name, top_label, top_label_score)
                image_url = upload_image_to_cloudinary(pil_image)
                emergency_level = get_emergency_level(top_label_score)
                # TODO: Add notification saving option

                send_crime_to_api(
                    camera_name=camera_name,
                    crime_type=crime_type_enum,
                    emergency_level=emergency_level,
                    image_url=image_url,
                    location_id=location_id,
                    location_name=location_name,
                )

                print(f"[{camera_name}] Top {k} crime types:")
                for i in range(k):
                    label = crime_type_model.config.id2label.get(top_indices[i].item(), "Unknown")
                    score = round(top_probs[i].item(), 3)
                    print(f"{label}: {score}")

            time.sleep(1 / 10)  # Avoid flooding CPU
        except Exception as e:
            print(f"❌ Error processing frame from {camera_name}: {e}")
            break

    cap.release()

def display_video():
    caps = [cv2.VideoCapture(cam["streamUrl"].strip()) for cam in real_cameras]
    while True:
        frames = [cap.read()[1] for cap in caps]
        for i, frame in enumerate(frames):
            if frame is not None:
                cv2.imshow(f"Camera {i}", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

# Start processing threads
threads = []
for camera in real_cameras:
    thread = threading.Thread(target=process_camera, args=(camera,))
    thread.start()
    threads.append(thread)

# Show live video
display_video()

# Wait for detection threads
for thread in threads:
    thread.join()
