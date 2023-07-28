import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.optimizers import Adam
from loss import FocalLoss

def siamese_resnet(input_shape, mask_shape, num_classes):
    base_model = ResNet101(weights='imagenet.h5', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    img_input_t0 = Input(shape=input_shape, name='img_input_t0')
    mask_input_t0 = Input(shape=mask_shape, name='mask_input_t0')
    img_input_t1 = Input(shape=input_shape, name='img_input_t1')
    mask_input_t1 = Input(shape=mask_shape, name='mask_input_t1')

    # Use ResNet101 for image feature extraction
    img_features_t0 = base_model(img_input_t0)
    img_features_t0 = tf.keras.layers.GlobalAveragePooling2D()(img_features_t0)

    img_features_t1 = base_model(img_input_t1)
    img_features_t1 = tf.keras.layers.GlobalAveragePooling2D()(img_features_t1)

    # Simple CNN for mask feature extraction
    mask_features_t0 = Conv2D(64, (3, 3), activation='relu', padding='same')(mask_input_t0)
    mask_features_t0 = MaxPooling2D((2, 2), padding='same')(mask_features_t0)
    mask_features_t0 = Conv2D(32, (3, 3), activation='relu', padding='same')(mask_features_t0)
    mask_features_t0 = MaxPooling2D((2, 2), padding='same')(mask_features_t0)
    mask_features_t0 = Flatten()(mask_features_t0)

    mask_features_t1 = Conv2D(64, (3, 3), activation='relu', padding='same')(mask_input_t1)
    mask_features_t1 = MaxPooling2D((2, 2), padding='same')(mask_features_t1)
    mask_features_t1 = Conv2D(32, (3, 3), activation='relu', padding='same')(mask_features_t1)
    mask_features_t1 = MaxPooling2D((2, 2), padding='same')(mask_features_t1)
    mask_features_t1 = Flatten()(mask_features_t1)

    # Concatenate image and mask features
    x = concatenate([img_features_t0, mask_features_t0, img_features_t1, mask_features_t1])

    # Dense layer for classification
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(8, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[img_input_t0, mask_input_t0, img_input_t1, mask_input_t1], outputs=[x])

    model.compile(loss=FocalLoss(alpha=0.25),
                  optimizer=Adam(learning_rate=1e-3),
                  metrics=['accuracy'])

    return model
