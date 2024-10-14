import cv2
import numpy as np

from src.utils.utils import normalize_path

# Define the image size
IMG_SIZE = (128, 128)
UMBRAL = 0.5

# Prediction
def predict_image_spam(model, image_path):
  image_path = normalize_path(image_path)
  img = cv2.imread(image_path)
  img = cv2.resize(img, IMG_SIZE)
  img = img / 255.0
  img = np.expand_dims(img, axis=0)  # Add batch dimension
  prediction = model.predict(img)
  return prediction > UMBRAL