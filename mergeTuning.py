import keras
import numpy as np
from keras.layers import *
from keras.models import Model, load_model
from util.modelUtils import word_acc
from NNModels import VGG, ResNet34, ResNet50, SEResNet50
from keras.utils import plot_model
import os
import config
from NNModels import *
from NNModels import MyModel

from util.modelUtils import word_acc
from keras.callbacks import EarlyStopping, LearningRateScheduler, CSVLogger, ModelCheckpoint, TensorBoard, \
    ReduceLROnPlateau
from keras.optimizers import *
from DataGenerator import DataGenerator
from argparse import ArgumentParser
import config


def buildModel(paths):
    """
    构建模型
    """
    modelList = []
    model_num = len(paths)
    char_num = 62

    for path in paths:
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

    """
    加权运算，输出结果
    """
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
    model = Model(input, out, name='mergedModel_' + model_num)
    return model


if __name__ == '__main__':
    """
    1. 读取配置文件
    2. 加载各个类型的配置数据
    """
    # model
    model_data = config.Model.model_data
    model_name = config.Model.model
    input_shape = config.Model.input_shape
    img_shape = config.Model.img_shape

    # train
    use_preweight = config.Train.use_preweight
    pretrained_weights = config.Train.pretrained_weights
    train_file = config.Train.train_data_file
    train_dir = config.Train.train_data_folder
    train_prob_from = config.Train.train_prob_from
    train_prob_to = config.Train.train_prob_to

    # hyper-parameter
    train_workers = config.Train.workers
    batch_size = config.Train.batch_size
    warmup_epochs = config.Train.warmup_epochs
    nb_epochs = config.Train.nb_epochs

    # valid
    valid_file = config.Valid.valid_data_file
    valid_dir = config.Valid.valid_data_folder
    valid_prob_from = config.Valid.valid_prob_from
    valid_prob_to = config.Valid.valid_prob_to

    # merge
    pathList = config.Merge.model_paths

    # 编译模型
    model = buildModel(pathList)
    model.compile(optimizer=Adam(0.001, amsgrad=True),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', word_acc])
    """
    创建数据生成器
    """
    # 训练数据生成器
    train_data_gen = DataGenerator(data_file=train_file, data_dir=train_dir, img_shape=img_shape, batch_size=batch_size,
                                   data_aug=True,
                                   prob_from=train_prob_from,
                                   prob_to=train_prob_to,
                                   shuffle=True)
    # 验证数据生成器
    valid_data_gen = DataGenerator(data_file=valid_file, data_dir=valid_dir, img_shape=img_shape, batch_size=batch_size,
                                   data_aug=False,
                                   prob_from=valid_prob_from,
                                   prob_to=valid_prob_to,
                                   shuffle=False)
    """
    正式训练
    """
    print('[INFO]  正式训练')
    # 创建保存训练结果的文件夹
    if not (os.path.exists(os.path.join(model_data, model.name, 'checkpoints'))):
        os.makedirs(os.path.join(model_data, model.name, 'checkpoints'))

    csvLogger = os.path.join(model_data, model.name, 'train_log.csv')
    tensorBoardLog = os.path.join(model_data, model.name, 'logs')
    # modelCheckPoint = "./model_data/model.{epoch:02d}-{val_loss:.8f}.h5"
    modelCheckPoint = os.path.join(model_data, model.name, 'checkpoints', model.name + '.h5')
    callbacks = [EarlyStopping(monitor='val_loss', patience=18),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=1),
                 CSVLogger(filename=csvLogger),
                 TensorBoard(log_dir=tensorBoardLog),
                 ModelCheckpoint(filepath=modelCheckPoint,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=2)]

    model.fit_generator(train_data_gen,
                        epochs=nb_epochs,
                        validation_data=valid_data_gen,
                        workers=train_workers,
                        use_multiprocessing=True,
                        callbacks=callbacks)
