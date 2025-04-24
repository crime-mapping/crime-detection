import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
from io import BytesIO
from PIL import Image

# Configuration       
cloudinary.config( 
    cloud_name = "dugocez66", 
    api_key = "828918583556253", 
    api_secret = "JT5wIOYL40WKOn49bdN34q-lAi8", 
    secure=True
)

def upload_image_to_cloudinary(pil_image, folder="crime_detection"):
    """Uploads a PIL image to Cloudinary and returns the secure URL."""
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    buffered.seek(0)

    upload_result = cloudinary.uploader.upload(buffered, folder=folder)
    return upload_result.get("secure_url")