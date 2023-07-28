import os
from tensorflow.keras.layers import *
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from loss import FocalLoss

def sinle_resnet(input_shape, num_classes):
    base_model = ResNet101(weights='imagenet.h5', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    img_input = Input(shape=input_shape, name='img_input')
    mask_input = Input(shape=(input_shape[0], input_shape[1], 1), name='mask_input')

    # Use ResNet101 for image feature extraction
    img_features = base_model(img_input)
    img_features = GlobalAveragePooling2D()(img_features)

    # Simple CNN for mask feature extraction
    mask_features = Conv2D(64, (3, 3), activation='relu', padding='same')(mask_input)
    mask_features = MaxPooling2D((2, 2), padding='same')(mask_features)
    mask_features = Conv2D(32, (3, 3), activation='relu', padding='same')(mask_features)
    mask_features = MaxPooling2D((2, 2), padding='same')(mask_features)
    mask_features = Flatten()(mask_features)

    # Concatenate image and mask features
    x = concatenate([img_features, mask_features])

    # Dense layer for classification
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(8, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[img_input, mask_input], outputs=[x])

    model.compile(loss=FocalLoss(alpha=0.25),
                  optimizer=Adam(learning_rate=1e-3),
                  metrics=['accuracy'])

    return model

