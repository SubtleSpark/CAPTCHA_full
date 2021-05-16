from keras.regularizers import l2
from keras.layers import *
from keras import Model
from .MyModel import MyModel


class SEResNet50(MyModel):
    def __init__(self, inputShape=(40, 120, 3), droprate=0.5, regularizer=0.01):
        super().__init__(inputShape=inputShape, droprate=droprate, regularizer=regularizer)

    """
    相较原始SEResNet50，每一层通道变为原来一半
    """
    def createModel(self):
        model_input = Input(shape=self.inputShape)
        identity_blocks = [3, 4, 6, 3]
        # Block 1
        layer = Conv2D(32, kernel_size=3, strides=1,
                       padding='same', kernel_initializer='he_normal',
                       use_bias=False)(model_input)
        layer = BatchNormalization()(layer)
        layer = ReLU()(layer)
        block_1 = MaxPooling2D(3, strides=2, padding='same')(layer)

        # Block 2
        block_2 = self.__conv_block(block_1, 32)
        block_2 = self.__squeeze_excitation_layer(block_2, out_dim=128, ratio=32.0, conv=True)
        for _ in range(identity_blocks[0] - 1):
            block_2 = self.__conv_block(block_1, 32)
            block_2 = self.__squeeze_excitation_layer(block_2, out_dim=128, ratio=32.0, conv=False)

        # Block 3
        block_3 = self.__conv_block(block_2, 64)
        block_3 = self.__squeeze_excitation_layer(block_3, out_dim=256, ratio=32.0, conv=True)
        for _ in range(identity_blocks[1] - 1):
            block_3 = self.__conv_block(block_2, 64)
            block_3 = self.__squeeze_excitation_layer(block_3, out_dim=256, ratio=32.0, conv=False)

        # Block 4
        block_4 = self.__conv_block(block_3, 128)
        block_4 = self.__squeeze_excitation_layer(block_4, out_dim=512, ratio=32.0, conv=True)
        for _ in range(identity_blocks[2] - 1):
            block_4 = self.__conv_block(block_3, 128)
            block_4 = self.__squeeze_excitation_layer(block_4, out_dim=512, ratio=32.0, conv=False)

        # Block 5
        block_5 = self.__conv_block(block_4, 256)
        block_5 = self.__squeeze_excitation_layer(block_5, out_dim=1024, ratio=32.0, conv=True)
        for _ in range(identity_blocks[2] - 1):
            block_5 = self.__conv_block(block_4, 256)
            block_5 = self.__squeeze_excitation_layer(block_5, out_dim=1024, ratio=32.0, conv=False)

        pooling = GlobalAveragePooling2D()(block_5)

        """
        添加 top 分类器
        """
        model_output = self.top(self.droprate, self.regularizer, pooling)
        model: Model = Model(inputs=model_input, outputs=model_output, name=self.__class__.__name__)

        return model

    def __squeeze_excitation_layer(self, input_layer, out_dim, ratio, conv):
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

    def __conv_block(self, input_layer, filters):
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
