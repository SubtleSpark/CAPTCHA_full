from keras import Input, Model
from keras.callbacks import EarlyStopping, CSVLogger
from keras.layers import BatchNormalization, Conv2D, LeakyReLU, MaxPooling2D, Flatten, Dropout, Dense, Concatenate, \
    Reshape, Add, Activation, AveragePooling2D, ZeroPadding2D
from keras.regularizers import l2
from keras import backend as K
import keras


def model(input_size, regularizer=0, droprate=0.5):
    return ResNet50(input_size, regularizer, droprate)


"""ResNet Begin"""


def ResNet50(input_size, regularizer=0, droprate=0.5):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model_data -- a Model() instance in Keras
    """
    print("[INFO] Using ResNet50")

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_size)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = __convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = __identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = __identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = __convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = __identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = __identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = __identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = __convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = __identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = __identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = __identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = __identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = __identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = __convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = __identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = __identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL
    X = AveragePooling2D((2, 2), name='avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dropout(rate=droprate)(X)
    X1 = Dense(62, kernel_regularizer=l2(regularizer), activation='softmax')(X)
    X2 = Dense(62, kernel_regularizer=l2(regularizer), activation='softmax')(X)
    X3 = Dense(62, kernel_regularizer=l2(regularizer), activation='softmax')(X)
    X4 = Dense(62, kernel_regularizer=l2(regularizer), activation='softmax')(X)
    X = Concatenate(axis=-1)([X1, X2, X3, X4])
    predicts = Reshape(target_shape=(4, 62))(X)
    # Create model_data
    model: Model = Model(inputs=X_input, outputs=predicts, name='ResNet50')

    return model


def __identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    ### END CODE HERE ###

    return X


def __convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # SHORTCUT PATH
    X_shortcut = Conv2D(F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1')(
        X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


"""ResNet End"""
if __name__ == '__main__':
    from keras.utils import plot_model

    nnm = model((128, 128, 3))
    nnm.summary()
    plot_model(model=nnm, to_file='../model_data/model_summary/' + nnm.name + '_Model.png', show_shapes=True)
