import keras
import tensorflow as tf
from keras.utils import plot_model


def squeeze_excitation_layer(input_layer, out_dim, ratio, conv):
    squeeze = tf.keras.layers.GlobalAveragePooling2D()(input_layer)

    excitation = tf.keras.layers.Dense(units=out_dim / ratio, activation='relu')(squeeze)
    excitation = tf.keras.layers.Dense(out_dim, activation='sigmoid')(excitation)
    excitation = tf.keras.layers.Reshape([1, 1, out_dim])(excitation)

    scale = tf.keras.layers.multiply([input_layer, excitation])

    if conv:
        shortcut = tf.keras.layers.Conv2D(out_dim, kernel_size=1, strides=1,
                                          padding='same', kernel_initializer='he_normal')(input_layer)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    else:
        shortcut = input_layer
    out = tf.keras.layers.add([shortcut, scale])
    return out


def conv_block(input_layer, filters):
    layer = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1,
                                   padding='same', kernel_initializer='he_normal',
                                   use_bias=False)(input_layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1,
                                   padding='same', kernel_initializer='he_normal',
                                   use_bias=False)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(filters * 4, kernel_size=1, strides=1,
                                   padding='same', kernel_initializer='he_normal',
                                   use_bias=False)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    return layer


def SE_ResNet50(input_shape):
    model_input = tf.keras.layers.Input(shape=input_shape)
    identity_blocks = [3, 4, 6, 3]
    # Block 1
    layer = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1,
                                   padding='same', kernel_initializer='he_normal',
                                   use_bias=False)(model_input)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    block_1 = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(layer)

    # Block 2
    block_2 = conv_block(block_1, 64)
    block_2 = squeeze_excitation_layer(block_2, out_dim=256, ratio=32.0, conv=True)
    for _ in range(identity_blocks[0] - 1):
        block_2 = conv_block(block_1, 64)
        block_2 = squeeze_excitation_layer(block_2, out_dim=256, ratio=32.0, conv=False)

    # Block 3
    block_3 = conv_block(block_2, 128)
    block_3 = squeeze_excitation_layer(block_3, out_dim=512, ratio=32.0, conv=True)
    for _ in range(identity_blocks[1] - 1):
        block_3 = conv_block(block_2, 128)
        block_3 = squeeze_excitation_layer(block_3, out_dim=512, ratio=32.0, conv=False)

    # Block 4
    block_4 = conv_block(block_3, 256)
    block_4 = squeeze_excitation_layer(block_4, out_dim=1024, ratio=32.0, conv=True)
    for _ in range(identity_blocks[2] - 1):
        block_4 = conv_block(block_3, 256)
        block_4 = squeeze_excitation_layer(block_4, out_dim=1024, ratio=32.0, conv=False)

    # Block 5
    block_5 = conv_block(block_4, 512)
    block_5 = squeeze_excitation_layer(block_5, out_dim=2048, ratio=32.0, conv=True)
    for _ in range(identity_blocks[2] - 1):
        block_5 = conv_block(block_4, 512)
        block_5 = squeeze_excitation_layer(block_5, out_dim=2048, ratio=32.0, conv=False)

    pooling = tf.keras.layers.GlobalAveragePooling2D()(block_5)
    model_output = tf.keras.layers.Dense(10, activation='softmax')(pooling)

    model = tf.keras.models.Model(model_input, model_output)
    return model


if __name__ == '__main__':
    nnm = SE_ResNet50((128, 128, 3))
    nnm.summary()
    plot_model(model=nnm, to_file='../model_data/SE_ResNet50_Model.png', show_shapes=True)
