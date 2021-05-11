from keras import Input, Model
import keras
from .MyModel import MyModel


class KerasResNet50(MyModel):
    def __init__(self, inputShape=(120, 40, 3), droprate=0.5, regularizer=0.001, weights='imagenet'):
        super().__init__(inputShape=inputShape, droprate=droprate, regularizer=regularizer)
        self.weights = weights
        self.model = self.createModel()

    def createModel(self):
        print("[INFO] Using KerasResNet50")
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
        model: Model = Model(inputs=input_tensor, outputs=model_output, name='KerasResNet50')

        return model

    def fixExceptTop(self):
        self.fix('global_average_pooling2d_1')


if __name__ == '__main__':
    nnm = KerasResNet50()
    nnm.fixExceptTop()
    model = nnm.getModel()
    for layer in model.layers:
        print(layer.name + "--->" + str(layer.trainable))
