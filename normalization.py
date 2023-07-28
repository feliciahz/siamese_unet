import os
import numpy as np
from PIL import Image

def normalize_and_resize(input_folder, output_folder, new_size):
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, file_name)

        if not os.path.isfile(input_file_path) or not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img = Image.open(input_file_path).convert('L')

        img = img.resize(new_size)

        img_data = np.asarray(img, dtype=np.float32)
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

        img_normalized = Image.fromarray(np.uint8(img_data * 255))

        output_file_name = os.path.splitext(file_name)[0] + '_normalized' + os.path.splitext(file_name)[1]
        output_file_path = os.path.join(output_folder, output_file_name)

        img_normalized.save(output_file_path)
