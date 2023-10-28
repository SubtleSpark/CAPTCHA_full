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

# valid
valid_file = config.Valid.valid_data_file
valid_dir = config.Valid.valid_data_folder
valid_prob_from = config.Valid.valid_prob_from
valid_prob_to = config.Valid.valid_prob_to

"""
这个模块专门用于训练SE模块，训练分为3步：
1. 只训练SE模块
2. 解冻分类器，继续训练
3. 解冻整个网络，继续训练
"""


def main():
    # 加载模型结构
    nnm: MyModel = None
    if model_name == 'SEResNet50':
        nnm = SEResNet50(inputShape=input_shape, regularizer=0.001, droprate=0.5)
    elif model_name == 'SEResNet50_h':
        nnm = SEResNet50_h(inputShape=input_shape, regularizer=0.001, droprate=0.5)
    else:
        print('[ERROR] model_name ' + model_name + ' not find')
        return

    model = nnm.getModel()

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

    # 训练前，创建保存训练结果的文件夹
    if not (os.path.exists(os.path.join(model_data, model.name, 'checkpoints'))):
        os.makedirs(os.path.join(model_data, model.name, 'checkpoints'))
    """
    训练：
    1. 只训练SE模块              epoch2
    2. 解冻分类器，继续训练        epoch2
    3. 解冻整个网络，继续训练      epoch3
    """
    epoch1 = 500
    epoch2 = 500
    epoch3 = 500

    """
    1. 只训练SE模块
    """
    print('[INFO]  step1. 只训练SE模块')

    # 加载权重
    model.load_weights(filepath=pretrained_weights, by_name=True, skip_mismatch=True)
    print('[INFO] using Weight ----> ' + pretrained_weights)

    # 指定要训练的层
    for layer in model.layers:
        if 'se_module' in str(layer.name):
            layer.trainable = True
            print('[INFO] ' + layer.name + ' is training')
        else:
            layer.trainable = False

    # 编译模型
    model.compile(optimizer=Adam(0.001, amsgrad=True),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', word_acc])

    csvLogger = os.path.join(model_data, model.name, 'tuning1' + 'train_log.csv')
    modelCheckPoint = os.path.join(model_data, model.name, 'checkpoints', model.name + 'tuning1' + '.h5')
    callbacks = [EarlyStopping(monitor='val_loss', patience=18),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=1),
                 CSVLogger(filename=csvLogger),
                 ModelCheckpoint(filepath=modelCheckPoint,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False)]

    model.fit(train_data_gen,
                        epochs=epoch1,
                        validation_data=valid_data_gen,
                        workers=train_workers,
                        use_multiprocessing=True,
                        callbacks=callbacks)
    """
    2. 解冻分类器，继续训练
    """
    print('[INFO]  2. 解冻分类器，继续训练')

    # 加载前一步最好的权重
    model.load_weights(filepath=modelCheckPoint, by_name=True, skip_mismatch=True)
    print('[INFO] using Weight ----> ' + modelCheckPoint)

    # 指定要训练的层
    for layer in model.layers:
        if 'se_module' in str(layer.name) or (layer.name in ('out0', 'out1', 'out2', 'out3')):
            layer.trainable = True
            print('[INFO] ' + layer.name + ' is training')
        else:
            layer.trainable = False

    # 编译模型
    model.compile(optimizer=Adam(0.001, amsgrad=True),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', word_acc])

    csvLogger = os.path.join(model_data, model.name, 'tuning2' + 'train_log.csv')
    modelCheckPoint = os.path.join(model_data, model.name, 'checkpoints', model.name + 'tuning2' + '.h5')
    callbacks = [EarlyStopping(monitor='val_loss', patience=18),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=1),
                 CSVLogger(filename=csvLogger),
                 ModelCheckpoint(filepath=modelCheckPoint,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 period=2)]

    model.fit_generator(train_data_gen,
                        epochs=epoch2,
                        validation_data=valid_data_gen,
                        workers=train_workers,
                        use_multiprocessing=True,
                        callbacks=callbacks)

    """
    3. 解冻整个网络，继续训练
    """
    print('[INFO]  3. 解冻整个网络，继续训练')

    # 加载前一步最好的权重
    model.load_weights(filepath=modelCheckPoint, by_name=True, skip_mismatch=True)
    print('[INFO] using Weight ----> ' + modelCheckPoint)

    # 指定要训练的层（全部训练）
    for layer in model.layers:
        layer.trainable = True

    # 编译模型
    model.compile(optimizer=Adam(0.001, amsgrad=True),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', word_acc])

    csvLogger = os.path.join(model_data, model.name, 'tuning3' + 'train_log.csv')
    modelCheckPoint = os.path.join(model_data, model.name, 'checkpoints', model.name + 'tuning3' + '.h5')
    callbacks = [EarlyStopping(monitor='val_loss', patience=18),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=1),
                 CSVLogger(filename=csvLogger),
                 ModelCheckpoint(filepath=modelCheckPoint,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 period=2)]

    model.fit_generator(train_data_gen,
                        epochs=epoch3,
                        validation_data=valid_data_gen,
                        workers=train_workers,
                        use_multiprocessing=True,
                        callbacks=callbacks)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model')
    parser.add_argument('-p', '--preweight')

    args = parser.parse_args()

    if args.model is not None:
        model_name = args.model
        print('[INFO] ArgumentParser: model = ' + args.model)
    if args.preweight is not None:
        pretrained_weights = args.preweight
        use_preweight = True
        print('[INFO] ArgumentParser: preweight = ' + args.preweight)

    main()
