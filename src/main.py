import os
from tensorboard.compat import tf
from src.utils.utils import get_images_root_folder, rename_td
from src.model.execute_model import predict_image_spam

# Edit the paths where the images are located and the model
root_folder = 'D:/Books/Source/manga/workspace/Arioto'
model_path = 'I:\Training\models'
model_name = 'remove_ads_v1.1.keras'

# Load model
model = tf.keras.models.load_model(os.path.join(model_path, model_name))

def check_by_image(image_list):
  for image in image_list:
    print(image)
    spam_prediction = predict_image_spam(model, image)
    if spam_prediction:
      rename_td(image)


def check_root_folder():
  image_list = get_images_root_folder(root_folder)
  check_by_image(image_list)


def main():
  print("Starting classification model...")
  check_root_folder()
  print("Finished classification model...")


main()
