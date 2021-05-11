from keras import Input, Model
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Concatenate, \
    Reshape, Activation
from .MyModel import MyModel


class LeNet(MyModel):

    def __init__(self, inputShape=(120, 40, 3), droprate=0.5, regularizer=0.01):
        super().__init__(inputShape=inputShape, droprate=droprate, regularizer=regularizer)
        self.model = self.createModel()

    def createModel(self):
        print("[INFO] Using LeNet")
        X_input = Input(shape=self.inputShape)
        X = X_input
        for i, n_cnn in enumerate([2, 2, 2, 2, 2]):
            for j in range(n_cnn):
                X = Conv2D(32 * 2 ** min(i, 3), kernel_size=3, padding='same')(X)
                X = BatchNormalization()(X)
                X = Activation('relu')(X)
            X = MaxPooling2D(2)(X)

        X = Flatten()(X)

        """
        添加 top 分类器
        """
        model_output = self.top(self.droprate, self.regularizer, X)
        model: Model = Model(X_input, model_output, name="LeNet")
        return model


if __name__ == '__main__':
    nnm = LeNet()
    nnm.showModelDetail()
