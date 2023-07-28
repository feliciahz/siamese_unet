import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import MeanIoU
from loss import FocalLoss, bce_dice_loss

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

def decoder_block(inputs, conv_from_encoder, filters):
    up = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(conv_from_encoder)
    up = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up)
    up = BatchNormalization()(up)
    inputs = UpSampling2D(size=(2, 2))(inputs)
    merge = concatenate([inputs, up], axis=3)
    conv = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
    conv = BatchNormalization()(conv)
    return conv

def single_multi_task(input_size, num_classes):
    inputs = Input(input_size)

    conv1, pool1 = encoder_block(inputs, 16)
    conv2, pool2 = encoder_block(pool1, 32)
    conv3, pool3 = encoder_block(pool2, 64)
    conv4, pool4 = encoder_block(pool3, 128)
    conv5 = encoder_block(pool4, 256, pool=False)

    conv5_up = UpSampling2D(size=(2, 2))(conv5)

    conv6 = decoder_block(conv5_up, conv4, 128)
    conv7 = decoder_block(conv6, conv3, 64)
    conv8 = decoder_block(conv7, conv2, 32)
    conv9 = decoder_block(conv8, conv1, 16)

    # Segmentation output
    segmentation = Conv2D(1, 1, activation='sigmoid', name='segmentation')(conv9)

    # Classification output
    gap3 = GlobalAveragePooling2D()(conv3)
    gap5 = GlobalAveragePooling2D()(conv5)
    gap9 = GlobalAveragePooling2D()(conv9)

    # Add the difference maps to the feature fusion
    fuse = concatenate([gap3, gap5, gap9])

    fc = Dense(32, activation='relu')(fuse)
    fc = Dropout(0.2)(fc)
    fc = Dense(8, activation='relu')(fc)
    fc = Dropout(0.2)(fc)
    mortality = Dense(num_classes, activation='softmax', name='mortality')(fc)

    # Define the model
    model = Model(inputs=[inputs], outputs=[segmentation, mortality])

    # Define the optimizer
    optimizer = Adam(learning_rate=1e-3)

    # Define metrics
    iou_metric = MeanIoU(num_classes=2)
    auc = tf.keras.metrics.AUC(name='auc')

    metrics = {'segmentation': iou_metric, 'mortality': ['accuracy', auc]}

    # Compile the model with separate losses for each output
    loss_weights = {'segmentation': 0.2, 'mortality': 1.}

    model.compile(optimizer=optimizer,
                  loss={'segmentation': bce_dice_loss, 'mortality': FocalLoss(alpha=0.25)},
                  loss_weights=loss_weights,
                  metrics=metrics)

    return model