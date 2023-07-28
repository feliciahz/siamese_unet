import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import Callback

class DelayedReduceLROnPlateau(Callback):
    def __init__(self, monitor='val_loss', factor=0.5, patience=0, verbose=0, min_lr=0, delay=0):
        super(DelayedReduceLROnPlateau, self).__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr
        self.wait = 0
        self.delay = delay
        self.best = 1e15

    def on_epoch_end(self, epoch, logs=None):
        if epoch <= self.delay:
            return

        current = logs.get(self.monitor)
        if current is None:
            return

        if current < self.best - self.min_lr:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                    if self.verbose > 0:
                        print(f'\nEpoch {epoch + 1}: reducing learning rate to {new_lr}.')
                self.wait = 0

class DelayedEarlyStopping(Callback):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, delay=0):
        super(DelayedEarlyStopping, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.delay = delay
        self.best = 1e15

    def on_epoch_end(self, epoch, logs=None):
        if epoch <= self.delay:
            return

        current = logs.get(self.monitor)
        if current is None:
            return

        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.verbose > 0:
                    print(f'\nEpoch {self.stopped_epoch + 1}: early stopping after {self.delay} epochs')

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f'\nEpoch {self.stopped_epoch + 1}: early stopping')
