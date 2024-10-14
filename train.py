import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


# Define the image size
IMG_SIZE = (128, 128)

# Define the paths to the folders
spam_path = 'I:\Training\manga_spam'
manga_path = 'I:\Training\manga_images'
save_model_path = 'I:\Training\models'

def load_images_from_folder(folder):
  images = []
  labels = []
  for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, filename))
    if img is not None:
      img = cv2.resize(img, IMG_SIZE)
      img = img / 255.0  # Normalize the image
      images.append(img)
      if 'spam' in folder:
        labels.append(1)  # Label spam/ad as 1
      else:
        labels.append(0)  # Label manga image as 0
  return np.array(images), np.array(labels)

spam_images, spam_labels = load_images_from_folder(spam_path)
manga_images, manga_labels = load_images_from_folder(manga_path)

# Combine spam and manga images
X = np.concatenate((spam_images, manga_images), axis=0)
y = np.concatenate((spam_labels, manga_labels), axis=0)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Building a simple CNN model
model = models.Sequential([
  layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(64, (3, 3), activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(128, (3, 3), activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Aqui voy

# Training the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

model.save(os.path.join(save_model_path, 'remove_ads_v1.1.keras'))