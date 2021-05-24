import keras
import numpy as np
from keras.layers import *
from keras.models import Model, load_model
from util.modelUtils import word_acc
from NNModels import VGG, ResNet34, ResNet50, SEResNet50
from keras.utils import plot_model

import config

pathList = config.Merge.model_paths
model_num = len(pathList)
char_num = 62

# 加载所有模型
modelList = []
for path in pathList:
    print('[INFO] loading model from ---->' + path)
    model: Model = load_model(path, custom_objects={"word_acc": word_acc})
    model.trainable = False
    modelList.append(model)

# 构建模型
input = Input(shape=config.Model.input_shape)
c0list = []
c1list = []
c2list = []
c3list = []
for model in modelList:
    x = Model(model.input, [model.get_layer('out0').output,
                            model.get_layer('out1').output,
                            model.get_layer('out2').output,
                            model.get_layer('out3').output])(input)
    c0list.append(x[0])
    c1list.append(x[1])
    c2list.append(x[2])
    c3list.append(x[3])

# 将各模型的各字符的预测结构进行连接
c0 = Concatenate()(c0list)
c1 = Concatenate()(c1list)
c2 = Concatenate()(c2list)
c3 = Concatenate()(c3list)
c0 = Reshape((model_num, char_num))(c0)  # 第0个字符各模型的预测结果
c1 = Reshape((model_num, char_num))(c1)  # 第1个字符各模型的预测结果
c2 = Reshape((model_num, char_num))(c2)  # 第2个字符各模型的预测结果
c3 = Reshape((model_num, char_num))(c3)  # 第3个字符各模型的预测结果

"""
生成权重
"""
w = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(input)
w = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(w)
a = GlobalAvgPool2D()(w)
m = GlobalMaxPool2D()(w)
w = Concatenate()([a, m])
w = Dense(model_num, activation='softmax')(w)

out0 = dot([c0, w, ], axes=(1, 1))
out1 = dot([c1, w, ], axes=(1, 1))
out2 = dot([c2, w, ], axes=(1, 1))
out3 = dot([c3, w, ], axes=(1, 1))
out0 = Activation('softmax')(out0)
out1 = Activation('softmax')(out1)
out2 = Activation('softmax')(out2)
out3 = Activation('softmax')(out3)

out = Concatenate()([out0, out1, out2, out3])
out = Reshape((4, 62))(out)
model = Model(input, out)

plot_model(model, to_file='./model_data/model_summary/merge.png', show_shapes=True)
model.summary()
