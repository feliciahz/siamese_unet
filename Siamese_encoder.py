import os
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from loss import FocalLoss

def encoder_block(inputs, filters, pool=True):
    conv = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal', strides=2)(inputs)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.2)(conv)
    conv = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    conv = BatchNormalization()(conv)
    if pool:
        next_layer = conv
        return conv, next_layer
    else:
        return conv


def tradition_siamese(input_size, num_classes):
    inputs1 = Input(input_size)
    inputs2 = Input(input_size)

    conv1_1, pool1_1 = encoder_block(inputs1, 16)
    conv1_2, pool1_2 = encoder_block(inputs2, 16)

    conv2_1, pool2_1 = encoder_block(pool1_1, 32)
    conv2_2, pool2_2 = encoder_block(pool1_2, 32)

    conv3_1, pool3_1 = encoder_block(pool2_1, 64)
    conv3_2, pool3_2 = encoder_block(pool2_2, 64)

    conv4_1, pool4_1 = encoder_block(pool3_1, 128)
    conv4_2, pool4_2 = encoder_block(pool3_2, 128)

    conv5_1 = encoder_block(pool4_1, 256, pool=False)
    conv5_2 = encoder_block(pool4_2, 256, pool=False)

    # Classification output
    concat5 = Concatenate()([conv5_1, conv5_2])
    gap5 = GlobalAveragePooling2D()(concat5)

    fc = Dense(32, activation='relu')(gap5)
    fc = Dropout(0.2)(fc)
    fc = Dense(8, activation='relu')(fc)
    fc = Dropout(0.2)(fc)
    mortality = Dense(num_classes, activation='softmax', name='mortality')(fc)

    # Define the model
    model = Model(inputs=[inputs1, inputs2], outputs=[mortality])

    # Define the optimizer
    optimizer = Adam(learning_rate=1e-3)

    # Define metrics
    auc = tf.keras.metrics.AUC(name='auc')
    metrics = {'mortality': ['accuracy', auc]}

    model.compile(optimizer=optimizer,
                  loss={'mortality': FocalLoss(alpha=0.25)},
                  metrics=metrics)

    return model


