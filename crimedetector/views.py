from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from .models import Camera

def get_active_cameras(request):
    cameras = Camera.objects.filter(active=True)
    camera_list = [{"id": cam.id, "name": cam.name, "source_url": cam.source_url} for cam in cameras]
    return JsonResponse({"cameras": camera_list})
