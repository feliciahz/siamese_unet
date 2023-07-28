import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.layers import Subtract
from loss import FocalLoss, bce_dice_loss

def encoder_block(inputs, filters, name, pool=True):
    conv = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal', strides=2, name=f"{name}_conv1")(inputs)
    conv = BatchNormalization(name=f"{name}_bn1")(conv)
    conv = Dropout(0.2, name=f"{name}_dropout1")(conv)
    conv = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal', name=f"{name}_conv2")(conv)
    conv = BatchNormalization(name=f"{name}_bn2")(conv)
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

def siamese_unet(input_size, num_classes):
    inputs1 = Input(input_size)
    inputs2 = Input(input_size)

    conv1_1, pool1_1 = encoder_block(inputs1, 16, 'conv1_1')
    conv1_2, pool1_2 = encoder_block(inputs2, 16, 'conv1_2')

    conv2_1, pool2_1 = encoder_block(pool1_1, 32, 'conv2_1')
    conv2_2, pool2_2 = encoder_block(pool1_2, 32, 'conv2_2')

    conv3_1, pool3_1 = encoder_block(pool2_1, 64, 'conv3_1')
    conv3_2, pool3_2 = encoder_block(pool2_2, 64, 'conv3_2')

    conv4_1, pool4_1 = encoder_block(pool3_1, 128, 'conv4_1')
    conv4_2, pool4_2 = encoder_block(pool3_2, 128, 'conv4_2')

    conv5_1 = encoder_block(pool4_1, 256, 'conv5_1', pool=False)
    conv5_2 = encoder_block(pool4_2, 256, 'conv5_2', pool=False)

    conv5_1_up = UpSampling2D(size=(2, 2))(conv5_1)
    conv5_2_up = UpSampling2D(size=(2, 2))(conv5_2)

    conv6_1 = decoder_block(conv5_1_up, conv4_1, 128)
    conv6_2 = decoder_block(conv5_2_up, conv4_2, 128)

    conv7_1 = decoder_block(conv6_1, conv3_1, 64)
    conv7_2 = decoder_block(conv6_2, conv3_2, 64)

    conv8_1 = decoder_block(conv7_1, conv2_1, 32)
    conv8_2 = decoder_block(conv7_2, conv2_2, 32)

    conv9_1 = decoder_block(conv8_1, conv1_1, 16)
    conv9_2 = decoder_block(conv8_2, conv1_2, 16)

    # Segmentation output
    segmentation1 = Conv2D(1, 1, activation='sigmoid', name='segmentation1')(conv9_1)
    segmentation2 = Conv2D(1, 1, activation='sigmoid', name='segmentation2')(conv9_2)

    # Classification output
    concat3 = Concatenate()([conv3_1, conv3_2])
    concat5 = Concatenate()([conv5_1, conv5_2])
    concat9 = Concatenate()([conv9_1, conv9_2])

    gap3 = GlobalAveragePooling2D()(concat3)
    gap5 = GlobalAveragePooling2D()(concat5)
    gap9 = GlobalAveragePooling2D()(concat9)

    diff3 = Subtract()([conv3_1, conv3_2])
    diff5 = Subtract()([conv5_1, conv5_2])
    diff9 = Subtract()([conv9_1, conv9_2])

    gap_diff3 = GlobalAveragePooling2D()(diff3)
    gap_diff5 = GlobalAveragePooling2D()(diff5)
    gap_diff9 = GlobalAveragePooling2D()(diff9)

    fuse = concatenate([gap3, gap5, gap9, gap_diff3, gap_diff5, gap_diff9])

    fc = Dense(32, activation='relu')(fuse)
    fc = Dropout(0.2)(fc)
    fc = Dense(8, activation='relu')(fc)
    fc = Dropout(0.2)(fc)
    mortality = Dense(num_classes, activation='softmax', name='mortality')(fc)

    # Define the model
    model = Model(inputs=[inputs1, inputs2], outputs=[segmentation1, segmentation2, mortality])

    # Define the optimizer
    optimizer = Adam(learning_rate=1e-3)

    # Define metrics
    iou_metric1 = MeanIoU(num_classes=2)
    iou_metric2 = MeanIoU(num_classes=2)
    auc = tf.keras.metrics.AUC(name='auc')

    metrics = {'segmentation1': iou_metric1, 'segmentation2': iou_metric2,
               'mortality': ['accuracy', auc]}

    # Compile the model with separate losses for each output
    loss_weights = {'segmentation1': 0.2, 'segmentation2': 0.2, 'mortality': 1.}

    model.compile(optimizer=optimizer,
                  loss={'segmentation1': bce_dice_loss, 'segmentation2': bce_dice_loss, 'mortality': FocalLoss(alpha=0.25)},
                  loss_weights=loss_weights,
                  metrics=metrics)

    return model
