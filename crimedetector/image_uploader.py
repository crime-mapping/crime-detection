import os
import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv


load_dotenv()

# Configuration       
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

def upload_image_to_cloudinary(pil_image, folder="crime_detection"):
    """Uploads a PIL image to Cloudinary and returns the secure URL."""
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    buffered.seek(0)

    upload_result = cloudinary.uploader.upload(buffered, folder=folder)
    return upload_result.get("secure_url")