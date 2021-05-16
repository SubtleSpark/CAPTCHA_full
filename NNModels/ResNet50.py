from keras import Input, Model
import keras
from .MyModel import MyModel


class ResNet50(MyModel):
    def __init__(self, inputShape=(40, 120, 3), droprate=0.5, regularizer=0.001, weights='imagenet'):
        self.weights = weights
        super().__init__(inputShape=inputShape, droprate=droprate, regularizer=regularizer)

    def createModel(self):
        input_tensor = Input(shape=self.inputShape)
        base_model = keras.applications.ResNet50(input_tensor=input_tensor,
                                                 include_top=False,
                                                 weights=self.weights,
                                                 pooling='avg')
        X = base_model.layers[-1].output

        """
        添加 top 分类器
        """
        model_output = self.top(self.droprate, self.regularizer, X)
        model: Model = Model(inputs=input_tensor, outputs=model_output, name=self.__class__.__name__)

        return model

    def fixExceptTop(self):
        self.fix('global_average_pooling2d_1')


