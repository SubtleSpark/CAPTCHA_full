from keras.regularizers import l2
from keras import Model, layers
from .MyModel import MyModel


class SEResNet50_h(MyModel):
    def __init__(self, inputShape=(40, 120, 3), droprate=0.5, regularizer=0.01):
        super().__init__(inputShape=inputShape, droprate=droprate, regularizer=regularizer)

    """
    相较原始SEResNet50，每一层通道变为原来一半
    """

    def createModel(self):
        model_input = layers.Input(shape=self.inputShape)
        x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(model_input)
        x = layers.Conv2D(64, (7, 7),
                          strides=(2, 2),
                          padding='valid',
                          kernel_initializer='he_normal',
                          name='conv1')(x)
        x = layers.BatchNormalization(name='bn_conv1')(x)
        x = layers.Activation('relu')(x)
        x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv_block(x, 3, [32, 32, 128], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [32, 32, 128], stage=2, block='b')
        x = identity_block(x, 3, [32, 32, 128], stage=2, block='c')

        x = conv_block(x, 3, [64, 64, 256], stage=3, block='a')
        x = identity_block(x, 3, [64, 64, 256], stage=3, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=3, block='c')
        x = identity_block(x, 3, [64, 64, 256], stage=3, block='d')

        x = conv_block(x, 3, [128, 128, 512], stage=4, block='a')
        x = identity_block(x, 3, [128, 128, 512], stage=4, block='b')
        x = identity_block(x, 3, [128, 128, 512], stage=4, block='c')
        x = identity_block(x, 3, [128, 128, 512], stage=4, block='d')
        x = identity_block(x, 3, [128, 128, 512], stage=4, block='e')
        x = identity_block(x, 3, [128, 128, 512], stage=4, block='f')

        x = conv_block(x, 3, [256, 256, 1024], stage=5, block='a')
        x = identity_block(x, 3, [256, 256, 1024], stage=5, block='b')
        x = identity_block(x, 3, [256, 256, 1024], stage=5, block='c')

        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

        """
        添加 top 分类器
        """
        model_output = self.top(self.droprate, self.regularizer, x)
        model: Model = Model(inputs=model_input, outputs=model_output, name=self.__class__.__name__)

        return model


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2c')(x)

    # se_module
    shape = x.get_shape().as_list()
    channel = shape[-1]

    squeeze = layers.GlobalAveragePooling2D()(x)

    excitation = layers.Dense(units=int(channel / 16), activation='relu')(squeeze)
    excitation = layers.Dense(channel, activation='sigmoid')(excitation)
    excitation = layers.Reshape([1, 1, channel])(excitation)
    se_module = layers.multiply([x, excitation])

    # add
    x = layers.add([se_module, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # conv
    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2c')(x)

    # shortcut
    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        name=bn_name_base + '1')(shortcut)

    # conv + shortcut
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x
