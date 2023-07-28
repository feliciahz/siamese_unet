import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from albumentations import *


def adjustData(img, mask):
    img = img / 255.
    mask = mask / 255.
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img, mask)


num_classes = 2

def trainGenerator(batch_size, train_images1, train_masks1, train_images2, train_masks2, labels,
                   image_color_mode="grayscale", mask_color_mode="grayscale", target_size=(256, 256)):

    labels = dict(zip(train_images1+train_images2, labels))  # updated to use full paths

    # Data augmentation configuration
    aug = Compose([HorizontalFlip(), VerticalFlip(), ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10), RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)])

    while True:
        for i in range(0, len(train_images1), batch_size):
            batch_images1 = train_images1[i:i+batch_size]
            batch_masks1 = train_masks1[i:i+batch_size]
            batch_images2 = train_images2[i:i+batch_size]
            batch_masks2 = train_masks2[i:i+batch_size]

            img1 = np.array([img_to_array(load_img(img_path, color_mode=image_color_mode, target_size=target_size)) for img_path in batch_images1])
            mask1 = np.array([img_to_array(load_img(mask_path, color_mode=mask_color_mode, target_size=target_size)) for mask_path in batch_masks1])
            img2 = np.array([img_to_array(load_img(img_path, color_mode=image_color_mode, target_size=target_size)) for img_path in batch_images2])
            mask2 = np.array([img_to_array(load_img(mask_path, color_mode=mask_color_mode, target_size=target_size)) for mask_path in batch_masks2])

            img1, mask1 = adjustData(img1, mask1)
            img2, mask2 = adjustData(img2, mask2)

            batch_labels = np.array([labels[img_path] for img_path in batch_images1], dtype=np.int32)
            batch_labels = to_categorical(batch_labels, num_classes=num_classes)

            # Apply data augmentation to the images and masks
            augmented_images1 = []
            augmented_masks1 = []
            augmented_images2 = []
            augmented_masks2 = []

            for img, msk in zip(img1, mask1):
                augmented = aug(image=img, mask=msk)
                augmented_images1.append(augmented['image'])
                augmented_masks1.append(augmented['mask'])

            for img, msk in zip(img2, mask2):
                augmented = aug(image=img, mask=msk)
                augmented_images2.append(augmented['image'])
                augmented_masks2.append(augmented['mask'])

            img1 = np.array(augmented_images1)
            mask1 = np.array(augmented_masks1)
            img2 = np.array(augmented_images2)
            mask2 = np.array(augmented_masks2)
            yield ([img1, img2], [mask1, mask2, batch_labels])


def valGenerator(batch_size, val_images1, val_masks1, val_images2, val_masks2, labels,
                   image_color_mode="grayscale", mask_color_mode="grayscale", target_size=(256, 256)):

    labels = dict(zip(val_images1+val_images2, labels))

    while True:
        for i in range(0, len(val_images1), batch_size):
            batch_images1 = val_images1[i:i+batch_size]
            batch_masks1 = val_masks1[i:i+batch_size]
            batch_images2 = val_images2[i:i+batch_size]
            batch_masks2 = val_masks2[i:i+batch_size]

            img1 = np.array([img_to_array(load_img(img_path, color_mode=image_color_mode, target_size=target_size)) for img_path in batch_images1])
            mask1 = np.array([img_to_array(load_img(mask_path, color_mode=mask_color_mode, target_size=target_size)) for mask_path in batch_masks1])
            img2 = np.array([img_to_array(load_img(img_path, color_mode=image_color_mode, target_size=target_size)) for img_path in batch_images2])
            mask2 = np.array([img_to_array(load_img(mask_path, color_mode=mask_color_mode, target_size=target_size)) for mask_path in batch_masks2])

            img1, mask1 = adjustData(img1, mask1)
            img2, mask2 = adjustData(img2, mask2)

            batch_labels = np.array([labels[img_path] for img_path in batch_images1], dtype=np.int32)
            batch_labels = to_categorical(batch_labels, num_classes=num_classes)

            yield ([img1, img2], [mask1, mask2, batch_labels])

