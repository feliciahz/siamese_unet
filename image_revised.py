import os
from PIL import Image

def crop_images(input_folder, output_folder, original_size, crop_size, crop_pixels):
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, file_name)

        if not os.path.isfile(input_file_path) or not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img = Image.open(input_file_path)

        if img.size != original_size:
            continue

        left = crop_pixels[0]
        top = crop_pixels[1]
        right = original_size[0] - crop_pixels[2]
        bottom = original_size[1] - crop_pixels[3]


        img_cropped = img.crop((left, top, right, bottom))

        output_file_name = os.path.splitext(file_name)[0] + '_cropped' + os.path.splitext(file_name)[1]
        output_file_path = os.path.join(output_folder, output_file_name)

        img_cropped.save(output_file_path)
