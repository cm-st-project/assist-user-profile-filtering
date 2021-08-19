import tensorflow as tf
import tensorflow_hub as hub

import requests
from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np

from nudenet import NudeDetector
import os

original_image_cache = {}


def preprocess_image(image):
    image = np.array(image)
    # reshape into shape [batch_size, height, width, num_channels]
    img_reshaped = tf.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    image = tf.image.convert_image_dtype(img_reshaped, tf.float32)
    return image


def load_image_from_url(url):
    """Returns an image with shape [1, height, width, num_channels]."""
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image = preprocess_image(image)
    return image


def load_image(image_url, image_size=256, dynamic_size=False, max_dynamic_size=512):
    """Loads and preprocesses images."""
    # Cache image file locally.
    if image_url in original_image_cache:
        img = original_image_cache[image_url]
    elif image_url.startswith('https://'):
        img = load_image_from_url(image_url)
    else:
        fd = tf.io.gfile.GFile(image_url, 'rb')
        img = preprocess_image(Image.open(fd))
    original_image_cache[image_url] = img
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img_raw = img
    if tf.reduce_max(img) > 1.0:
        img = img / 255.
    if len(img.shape) == 3:
        img = tf.stack([img, img, img], axis=-1)
    if not dynamic_size:
        img = tf.image.resize_with_pad(img, image_size, image_size)
    elif img.shape[1] > max_dynamic_size or img.shape[2] > max_dynamic_size:
        img = tf.image.resize_with_pad(img, max_dynamic_size, max_dynamic_size)
    return img, img_raw


class_names = ['low', 'mid', 'high']


# return a score of 0 to 100
def evaluate_image(url, image_model, nude_model):
    excuse = ''
    score = 0
    image, _ = load_image(url, image_size=384)

    # neural network
    prediction_scores = image_model.predict(image)
    predicted_index = np.argmax(prediction_scores)
    scalar_cap = 2.0
    predicted_value = prediction_scores[0][predicted_index]
    if predicted_value > scalar_cap:
        predicted_value = scalar_cap
    print("predicted_value", predicted_value)
    print("Predicted label: " + class_names[predicted_index])
    print("predicted score: ", prediction_scores)
    if class_names[predicted_index] == 'low':
        score = 30 * predicted_value / scalar_cap
    elif class_names[predicted_index] == 'mid':
        score = 30 + predicted_value / scalar_cap * (60 - 30)
    else:
        score = 60 + predicted_value / scalar_cap * (100 - 60)

    # Classify single image
    res = nude_model.detect(url, mode='fast')

    # if no face detected, reduce the score
    if len(res) == 0:
        if score > 30:
            score = score - 30
        excuse = 'Please use a picture of yourself.'
    elif res[0]['label'] == 'FACE_F' or res[0]['label'] == 'FACE_M':
        if score < 30:
            score = score + 30
        print("detected " + res[0]['label'])
    else:
        score = 0
        excuse = ("Detected inappropriate content:" + res[0]['label'])

    if score < 0:
        score = 0
    elif score > 100:
        score = 100
    else:
        score = round(score)

    return score, excuse
