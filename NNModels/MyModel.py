from keras import Input, Model
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Concatenate, \
    Reshape, Activation
from keras.regularizers import l2
from keras.utils import plot_model
import abc


class MyModel:
    """
    构建自己的模型，主要定义了模型的 top 分类器。
    1. 继承这个类
    2. 重写 createModel 方法。 方法返回一个 keras.Model
    3. 在子类的 __init__ 方法中调用：
        self.model = self.createModel()
    """

    def __init__(self, inputShape, droprate, regularizer):
        self.inputShape = inputShape
        self.droprate = droprate
        self.regularizer = regularizer
        self.model: Model = None

    def getModel(self):
        return self.model

    @abc.abstractmethod
    def createModel(self):
        pass

    def top(self, droprate, regularizer, input):
        # output layer
        X = Dropout(rate=droprate)(input)
        X1 = Dense(62, kernel_regularizer=l2(regularizer), activation='softmax', name='out0')(X)
        X2 = Dense(62, kernel_regularizer=l2(regularizer), activation='softmax', name='out1')(X)
        X3 = Dense(62, kernel_regularizer=l2(regularizer), activation='softmax', name='out2')(X)
        X4 = Dense(62, kernel_regularizer=l2(regularizer), activation='softmax', name='out3')(X)
        X = Concatenate(axis=-1)([X1, X2, X3, X4])
        model_output = Reshape(target_shape=(4, 62))(X)

        return model_output

    def showModelDetail(self):
        if (self.model == None):
            print("[ERROE] model is None")
            return

        self.model.summary()
        plot_model(model=self.model, to_file='../model_data/model_summary/' + self.model.name + '_Model.png',
                   show_shapes=True)

    def fixAll(self):
        """
        冻结模型全部层
        """
        if (self.model == None):
            print("[ERROE] model is None")
            return

        for layer in self.model.layers:
            layer.trainable = False

    def fix(self, toLayer):
        """
            从第1层 ~ toLayer层冻结，停止训练
            toLayer为某一层的name
            """
        for layer in self.model.layers:
            layer.trainable = False
            if (layer.name == toLayer):
                break

    def unfixAll(self):
        """
        解冻全部层
        """
        if (self.model == None):
            print("[ERROE] model is None")
            return

        for layer in self.model.layers:
            layer.trainable = True
