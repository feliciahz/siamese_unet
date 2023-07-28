from tensorflow.keras import backend as K
import tensorflow as tf

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2., alpha=0.25,
                 reduction=tf.keras.losses.Reduction.AUTO, name='focal_loss'):

        super(FocalLoss, self).__init__(reduction=reduction,
                                        name=name)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def call(self, y_true, y_pred):

        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(
            tf.subtract(1., model_out), self.gamma))
        fl = tf.multiply(self.alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)

def bce_dice_loss(y_true, y_pred, alpha=1, beta=1):
    bce_loss = K.binary_crossentropy(y_true, y_pred)
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    dice_loss = 1 - (2 * intersection + 1) / (union + 1)
    return alpha * bce_loss + beta * dice_loss