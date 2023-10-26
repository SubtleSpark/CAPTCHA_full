import keras.backend as K
from keras import Model
from keras.metrics import categorical_accuracy


def word_acc(y_true, y_pred):
    out = categorical_accuracy(y_true=y_true, y_pred=y_pred)
    return K.min(out, axis=-1)


def fix_all(model: Model):
    for layer in model.layers:
        layer.trainable = False


def unfix_all(model: Model):
    for layer in model.layers:
        layer.trainable = True
