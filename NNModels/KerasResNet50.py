from keras import Input, Model
from keras.layers import BatchNormalization, Conv2D, LeakyReLU, MaxPooling2D, Flatten, Dropout, Dense, Concatenate, \
    Reshape, Add, Activation, AveragePooling2D, ZeroPadding2D
from keras.regularizers import l2
import keras


def model(input_size, regularizer=0, droprate=0.5, weights='imagenet'):
    return KerasResNet50(input_size, regularizer, droprate, weights)


def KerasResNet50(input_size, regularizer=0, droprate=0.5, weights='imagenet'):
    print("[INFO] Using KerasResNet50")
    # input_tensor = Input(shape=(224, 224, 3))
    input_tensor = Input(shape=input_size)
    base_model = keras.applications.ResNet50(input_tensor=input_tensor,
                                             include_top=False,
                                             weights=weights,
                                             pooling='avg')
    # X = K.function([base_model.layers[0].input, K.learning_phase()], [base_model.layers[-1].output]).outputs
    X = base_model.layers[-1].output

    # output layer
    X = Dropout(rate=droprate)(X)
    X1 = Dense(62, kernel_regularizer=l2(regularizer), activation='softmax')(X)
    X2 = Dense(62, kernel_regularizer=l2(regularizer), activation='softmax')(X)
    X3 = Dense(62, kernel_regularizer=l2(regularizer), activation='softmax')(X)
    X4 = Dense(62, kernel_regularizer=l2(regularizer), activation='softmax')(X)
    X = Concatenate(axis=-1)([X1, X2, X3, X4])
    predicts = Reshape(target_shape=(4, 62))(X)

    # Create model_data
    model: Model = Model(inputs=input_tensor, outputs=predicts, name='KerasResNet50')

    return model


"""
从第1层 ~ toLayer层冻结（不冻结toLayer），停止训练
toLayer为某一层的name
"""


def fix(base_model: Model, toLayer):
    for layer in base_model.layers:
        if (layer.name == toLayer):
            return base_model
        else:
            layer.trainable = False


def fixExceptTop(model: Model):
    fix(model, 'global_average_pooling2d_1')


if __name__ == '__main__':
    KResNet = model((128, 128, 3), weights=None)
    fix(KResNet, 'global_average_pooling2d_1')
    for layer in KResNet.layers:
        print(layer.name, ' is trainable? ', layer.trainable)
