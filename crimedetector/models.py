from django.db import models

from django.db import models
from django.contrib.auth.models import User

class Camera(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="cameras")
    name = models.CharField(max_length=255)  # Camera name
    source_url = models.CharField(max_length=500)  # RTSP/IP/Local Source
    active = models.BooleanField(default=True)  # Is the camera enabled?
    email_notifications = models.BooleanField(default=True)  # Send alerts?

    def __str__(self):
        return f"{self.name} - {self.source_url}"

