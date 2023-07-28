import os
import pandas as pd
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from sklearn.model_selection import train_test_split
from callbacks import DelayedReduceLROnPlateau, DelayedEarlyStopping
from Siamese_Unet import siamese_unet
from generator import trainGenerator, valGenerator


train_dir = os.path.join(base_dir, 'train')

image_folder = 'images'
mask_folder = 'masks'
input_size = (256, 256, 1)
num_classes = 2

# Create the model
model = siamese_unet(input_size, num_classes)


# Split the data into train and validation sets
def split_data(train_dir, test_size=0.2, random_state=5):
    # Get the image and mask directories for time point 0
    image_dir1 = os.path.join(train_dir, 't0', 'images')
    mask_dir1 = os.path.join(train_dir, 't0', 'masks')

    # Get the image and mask directories for time point 1
    image_dir2 = os.path.join(train_dir, 't1', 'images')
    mask_dir2 = os.path.join(train_dir, 't1', 'masks')

    # Get the labels file
    labels_file = os.path.join(train_dir, 'labels.csv')

    # Read the labels file
    labels_df = pd.read_csv(labels_file)
    labels_df.set_index('image', inplace=True)
    labels = labels_df['label']

    # Get the list of image and mask files for time point 0
    images1 = [os.path.join(image_dir1, img) for img in os.listdir(image_dir1)]
    masks1 = [os.path.join(mask_dir1, mask) for mask in os.listdir(mask_dir1)]

    # Get the list of image and mask files for time point 1
    images2 = [os.path.join(image_dir2, img) for img in os.listdir(image_dir2)]
    masks2 = [os.path.join(mask_dir2, mask) for mask in os.listdir(mask_dir2)]

    # Split the data into train and validation sets
    train_images1, val_images1, train_masks1, val_masks1, train_labels, val_labels = train_test_split(
        images1, masks1, labels, test_size=test_size, random_state=random_state)

    train_images2, val_images2, train_masks2, val_masks2 = train_test_split(
        images2, masks2, test_size=test_size, random_state=random_state)

    return (train_images1, train_masks1, train_images2, train_masks2, train_labels), (
        val_images1, val_masks1, val_images2, val_masks2, val_labels)

# Split the data into train and validation sets
(train_images1, train_masks1, train_images2, train_masks2, train_labels), (val_images1, val_masks1, val_images2, val_masks2, val_labels) = split_data(train_dir)


reduce_lr = DelayedReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, delay=10)
early_stopping = DelayedEarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, delay=10)
callbacks = [reduce_lr, early_stopping]

# Create the data generators
batch_size = 32
train_gen = trainGenerator(batch_size, train_images1, train_masks1, train_images2, train_masks2, train_labels)
val_gen = valGenerator(batch_size, val_images1, val_masks1, val_images2, val_masks2, val_labels)

# Train the model
epochs = 100
steps_per_epoch = len(train_images1) // batch_size
validation_steps = len(val_images1) // batch_size


model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs,
          validation_data=val_gen, validation_steps=validation_steps,
          callbacks=callbacks)
