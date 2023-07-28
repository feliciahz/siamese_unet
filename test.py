import os
import csv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.io import imsave
from loss import FocalLoss, bce_dice_loss

predictions1 = {}
predictions2 = {}
mortality_preds_dict = {}

thresh = 0.5

# Iterate over each pair of images
for img_name1, img_name2 in zip(test_images1, test_images2):
    img_path1 = os.path.join(test_dir1, img_name1)
    img_path2 = os.path.join(test_dir2, img_name2)

    # Load the two images and convert to arrays
    img1 = load_img(img_path1, color_mode='grayscale', target_size=(256, 256))
    img1 = img_to_array(img1).astype('float32') / 255.

    img2 = load_img(img_path2, color_mode='grayscale', target_size=(256, 256))
    img2 = img_to_array(img2).astype('float32') / 255.

    # Reshape to match model's expected input shape
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)

    # Make prediction
    seg_preds1, seg_preds2, mortality_preds = model.predict([img1, img2])

    # Store the predictions
    patient_name1 = img_name1.split('.')[0]  # assuming the name is before the dot in the image file name
    patient_name2 = img_name2.split('.')[0]
    predictions1[patient_name1] = (seg_preds1[0, :, :, 0] > thresh).astype(np.uint8)
    predictions2[patient_name2] = (seg_preds2[0, :, :, 0] > thresh).astype(np.uint8)
    mortality_preds_dict[patient_name1] = mortality_preds[0]

for patient_name, prediction in predictions1.items():
    # Save the segmentation prediction as an image
    img = (prediction * 255).astype(np.uint8)
    imsave(os.path.join(predict_dir1, f"{patient_name}_predict.jpg"), img)

for patient_name, prediction in predictions2.items():
    # Save the segmentation prediction as an image
    img = (prediction * 255).astype(np.uint8)
    imsave(os.path.join(predict_dir2, f"{patient_name}_predict.jpg"), img)

# Save the mortality predictions to a CSV file
with open('siamese_unet_mortality_predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['PatientName',  'MortalityPrediction_1'])
    for patient_name, mortality_pred in mortality_preds_dict.items():
        writer.writerow([patient_name,  mortality_pred[1]])
