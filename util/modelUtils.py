import keras.backend as K
from keras.metrics import categorical_accuracy


def word_acc(y_true, y_pred):
    out = categorical_accuracy(y_true=y_true, y_pred=y_pred)
    return K.min(out, axis=-1)
