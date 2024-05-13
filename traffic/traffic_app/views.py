from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import default_storage

import os
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
IMG_WIDTH = 30
IMG_HEIGHT = 30

# Load your saved model
model = load_model('model.h5')

# Initializing class names
class_names = []
for i in range(43):
    class_names.append(f"Class: {i} ")



# Create your views here.
def index(request):
    if request.method == "POST":
        image = request.FILES['image']
        with open('traffic_app/image.jpg', 'wb') as f:
            f.write(image.file.read())
        
        file_name = 'image.jpg'
        file_path = default_storage.path(file_name)
        file_url = default_storage.url(file_name)
        print(file_path)
        img = tf.keras.utils.load_img(file_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
        return render(request,"traffic_app/index.html",{
            "class": class_names[np.argmax(score)],
            "file_url":file_url
        })
    return render(request,"traffic_app/index.html",{
        "class": "None"
    })