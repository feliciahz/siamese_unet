from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from loss import bce_dice_loss
from tensorflow.keras.metrics import MeanIoU

def encoder_block(input_layer, filters):
    conv1 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_layer)
    conv1 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool = MaxPooling2D(pool_size=(2, 2))(conv1)
    return conv1, pool

def decoder_block(input_layer, skip_features, filters):
    up = Conv2D(filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(input_layer))
    merge = concatenate([skip_features, up], axis=3)
    conv = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
    conv = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    return conv

def unet(input_size):
    inputs = Input(input_size)

    # Encoder
    conv1, pool1 = encoder_block(inputs, 16)
    conv2, pool2 = encoder_block(pool1, 32)
    conv3, pool3 = encoder_block(pool2, 64)
    conv4, pool4 = encoder_block(pool3, 128)

    # Center
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.2)(conv5)

    # Decoder
    conv6 = decoder_block(drop5, conv4, 128)
    conv7 = decoder_block(conv6, conv3, 64)
    conv8 = decoder_block(conv7, conv2, 32)
    conv9 = decoder_block(conv8, conv1, 16)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    # Define metrics
    iou_metric = MeanIoU(num_classes=2)

    model.compile(optimizer=Adam(learning_rate=1e-3), loss=bce_dice_loss, metrics=iou_metric)

    return model
