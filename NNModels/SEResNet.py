import keras
import tensorflow as tf
from keras.utils import plot_model
from keras.regularizers import l2
from keras.layers import *
from keras import Model


def model(input_size, regularizer=0.001, droprate=0.5):
    return SEResNet50(input_size, regularizer, droprate)


def SEResNet50(input_shape, regularizer, droprate):
    print("[INFO] Using SEResNet50")

    model_input = Input(shape=input_shape)
    identity_blocks = [3, 4, 6, 3]
    # Block 1
    layer = Conv2D(64, kernel_size=3, strides=1,
                   padding='same', kernel_initializer='he_normal',
                   use_bias=False)(model_input)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    block_1 = MaxPooling2D(3, strides=2, padding='same')(layer)

    # Block 2
    block_2 = __conv_block(block_1, 64)
    block_2 = __squeeze_excitation_layer(block_2, out_dim=256, ratio=32.0, conv=True)
    for _ in range(identity_blocks[0] - 1):
        block_2 = __conv_block(block_1, 64)
        block_2 = __squeeze_excitation_layer(block_2, out_dim=256, ratio=32.0, conv=False)

    # Block 3
    block_3 = __conv_block(block_2, 128)
    block_3 = __squeeze_excitation_layer(block_3, out_dim=512, ratio=32.0, conv=True)
    for _ in range(identity_blocks[1] - 1):
        block_3 = __conv_block(block_2, 128)
        block_3 = __squeeze_excitation_layer(block_3, out_dim=512, ratio=32.0, conv=False)

    # Block 4
    block_4 = __conv_block(block_3, 256)
    block_4 = __squeeze_excitation_layer(block_4, out_dim=1024, ratio=32.0, conv=True)
    for _ in range(identity_blocks[2] - 1):
        block_4 = __conv_block(block_3, 256)
        block_4 = __squeeze_excitation_layer(block_4, out_dim=1024, ratio=32.0, conv=False)

    # Block 5
    block_5 = __conv_block(block_4, 512)
    block_5 = __squeeze_excitation_layer(block_5, out_dim=2048, ratio=32.0, conv=True)
    for _ in range(identity_blocks[2] - 1):
        block_5 = __conv_block(block_4, 512)
        block_5 = __squeeze_excitation_layer(block_5, out_dim=2048, ratio=32.0, conv=False)

    pooling = GlobalAveragePooling2D()(block_5)

    # output layer
    X = Dropout(rate=droprate)(pooling)
    X1 = Dense(62, kernel_regularizer=l2(regularizer), activation='softmax')(X)
    X2 = Dense(62, kernel_regularizer=l2(regularizer), activation='softmax')(X)
    X3 = Dense(62, kernel_regularizer=l2(regularizer), activation='softmax')(X)
    X4 = Dense(62, kernel_regularizer=l2(regularizer), activation='softmax')(X)
    X = Concatenate(axis=-1)([X1, X2, X3, X4])
    model_output = Reshape(target_shape=(4, 62))(X)

    model = Model(model_input, model_output)
    return model


def __squeeze_excitation_layer(input_layer, out_dim, ratio, conv):
    squeeze = GlobalAveragePooling2D()(input_layer)

    excitation = Dense(units=int(out_dim / ratio), activation='relu')(squeeze)
    excitation = Dense(out_dim, activation='sigmoid')(excitation)
    excitation = Reshape([1, 1, out_dim])(excitation)

    scale = multiply([input_layer, excitation])

    if conv:
        shortcut = Conv2D(out_dim, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(
            input_layer)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = input_layer
    out = add([shortcut, scale])
    return out


def __conv_block(input_layer, filters):
    layer = Conv2D(filters, kernel_size=1, strides=1,
                   padding='same', kernel_initializer='he_normal',
                   use_bias=False)(input_layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    layer = Conv2D(filters, kernel_size=3, strides=1,
                   padding='same', kernel_initializer='he_normal',
                   use_bias=False)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    layer = Conv2D(filters * 4, kernel_size=1, strides=1,
                   padding='same', kernel_initializer='he_normal',
                   use_bias=False)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    return layer


if __name__ == '__main__':
    nnm = model((128, 128, 3))
    nnm.summary()
    plot_model(model=nnm, to_file='../model_data/SE_ResNet50_Model.png', show_shapes=True)
