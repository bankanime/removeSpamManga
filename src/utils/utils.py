import os
import unicodedata

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def get_image_paths(root_folder):
  image_paths = []

  for dirpath, _, filenames in os.walk(root_folder):
    for filename in filenames:
      if filename.lower().endswith(VALID_EXTENSIONS):
        image_paths.append(os.path.join(dirpath, filename))

  return image_paths

def get_images_root_folder(root_folder):
  image_paths = get_image_paths(root_folder)
  print(f"Found {len(image_paths)} images.")
  return image_paths

# ----------------------------
def rename_td(path_file):
  base, ext = os.path.splitext(path_file)
  new_name = f"{base}_td{ext}"
  new_path = os.path.join(os.path.dirname(path_file), new_name)
  os.rename(path_file, new_path)

# ----------------------------
def normalize_path(path):
  # Replace forward slashes with backslashes
  path = path.replace('/', '\\')

  # Normalize the path to remove accents
  normalized_path = unicodedata.normalize('NFD', path)
  normalized_path = ''.join(c for c in normalized_path if unicodedata.category(c) != 'Mn')

  return normalized_path