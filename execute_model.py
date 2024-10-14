import os
import cv2
import numpy as np
from tensorboard.compat import tf

save_model_path = 'I:\Training\models'

model = tf.keras.models.load_model(os.path.join(save_model_path, 'remove_ads_v1.keras'))

# Define the image size
IMG_SIZE = (128, 128)

# Prediction
def predict_image(image_path):
  img = cv2.imread(image_path)
  img = cv2.resize(img, IMG_SIZE)
  img = img / 255.0
  img = np.expand_dims(img, axis=0)  # Add batch dimension
  prediction = model.predict(img)
  return "Spam/Ad" if prediction > 0.5 else "Manga Image"

# Example usage
result = predict_image('D:/Books/Source/manga/workspace/Arioto/Capitulo001/031.png')
print(f"The image is: {result}")