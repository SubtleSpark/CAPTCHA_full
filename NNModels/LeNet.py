from keras import Input, Model
from keras.callbacks import EarlyStopping, CSVLogger
from keras.layers import BatchNormalization, Conv2D, LeakyReLU, MaxPooling2D, Flatten, Dropout, Dense, Concatenate, \
    Reshape, Add, Activation, AveragePooling2D, ZeroPadding2D
from keras.regularizers import l2
from keras import backend as K
import keras


def model(input_size, regularizer=0, droprate=0):
    return LeNet(input_size, regularizer, droprate)


def LeNet(input_size, regularizer=0, droprate=0):
    print("[INFO] Using LeNet")
    X_input = Input(shape=input_size)
    X = X_input
    for i, n_cnn in enumerate([2, 2, 2, 2, 2]):
        for j in range(n_cnn):
            X = Conv2D(32 * 2 ** min(i, 3), kernel_size=3, padding='same')(X)
            X = BatchNormalization()(X)
            X = Activation('relu')(X)
        X = MaxPooling2D(2)(X)

    X = Flatten()(X)
    X = Dropout(rate=droprate)(X)
    X1 = Dense(62, kernel_regularizer=l2(regularizer), activation='softmax')(X)
    X2 = Dense(62, kernel_regularizer=l2(regularizer), activation='softmax')(X)
    X3 = Dense(62, kernel_regularizer=l2(regularizer), activation='softmax')(X)
    X4 = Dense(62, kernel_regularizer=l2(regularizer), activation='softmax')(X)
    X = Concatenate(axis=-1)([X1, X2, X3, X4])
    pred = Reshape(target_shape=(4, 62))(X)

    model: Model = Model(X_input, pred, name="LeNet")
    return model
