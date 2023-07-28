import os
import nrrd
import numpy as np
from PIL import Image

for filename in os.listdir(input_image_folder):
    if filename.endswith(".nrrd"):

        input_image_path = os.path.join(input_image_folder, filename)
        image_data, _ = nrrd.read(input_image_path)

        image_data_2d = image_data[:, :, 0].T
        image = Image.fromarray(np.uint8(image_data_2d))
        image = image.resize(target_size)

        output_image_path = os.path.join(output_image_folder, filename[:-5] + ".jpg")
        image.save(output_image_path, "JPEG")

        input_mask_path = os.path.join(input_mask_folder, filename)
        mask_data, _ = nrrd.read(input_mask_path)

        mask_data_2d = mask_data[:, :, 0].T
        mask = Image.fromarray(np.uint8(mask_data_2d * 255))
        mask = mask.resize(target_size)

        output_mask_path = os.path.join(output_mask_folder, filename[:-5] + ".jpg")
        mask.save(output_mask_path, "JPEG")
