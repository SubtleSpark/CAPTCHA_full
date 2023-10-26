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
train_dir = config.Train.train_data_folder
train_prob_from = config.Train.train_prob_from
train_prob_to = config.Train.train_prob_to

# hyper-parameter
train_workers = config.Train.workers
batch_size = config.Train.batch_size
warmup_epochs = config.Train.warmup_epochs
nb_epochs = config.Train.nb_epochs

# valid
valid_dir = config.Valid.valid_data_folder
valid_prob_from = config.Valid.valid_prob_from
valid_prob_to = config.Valid.valid_prob_to


# from .VGG import VGG
# from .ResNet34 import ResNet34
# from .ResNet50 import ResNet50
# from .KerasResNet50 import KerasResNet50
# from .SEResNet50 import SEResNet50
# from .SEResNet50_h import SEResNet50_h

def main():
    # 加载模型结构
    nnm: MyModel = None
    if model_name == 'VGG':
        nnm = VGG(inputShape=input_shape, regularizer=0.001, droprate=0.5)
    elif model_name == 'ResNet34':
        nnm = ResNet34(inputShape=input_shape, regularizer=0.001, droprate=0.5)
    elif model_name == 'ResNet50':
        nnm = ResNet50(inputShape=input_shape, regularizer=0.001, droprate=0.5)
    elif model_name == 'KerasResNet50':
        nnm = KerasResNet50(inputShape=input_shape, regularizer=0.001, droprate=0.5)
    elif model_name == 'SEResNet50':
        nnm = SEResNet50(inputShape=input_shape, regularizer=0.001, droprate=0.5)
    elif model_name == 'SEResNet50_h':
        nnm = SEResNet50_h(inputShape=input_shape, regularizer=0.001, droprate=0.5)
    else:
        print('[ERROR] model_name ' + model_name + 'not find')

    model = nnm.getModel()

    """
    根据参数加载模型数据
    """
    # 读取权重
    if use_preweight:
        model.load_weights(filepath=pretrained_weights, by_name=True, skip_mismatch=True)
        print('[INFO] using Weight ----> ' + pretrained_weights)

    # 编译模型
    model.compile(optimizer=Adam(0.001, amsgrad=True),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', word_acc])

    """
    创建数据生成器
    """
    # 训练数据生成器
    train_data_gen = DataGenerator(data_dir=train_dir, img_shape=img_shape, batch_size=batch_size,
                                   data_aug=True,
                                   prob_from=train_prob_from,
                                   prob_to=train_prob_to,
                                   shuffle=True)
    # 验证数据生成器
    valid_data_gen = DataGenerator(data_dir=valid_dir, img_shape=img_shape, batch_size=batch_size,
                                   data_aug=False,
                                   prob_from=valid_prob_from,
                                   prob_to=valid_prob_to,
                                   shuffle=False)

    """
    训练：
    1. warmup
    2. 正式训练
    """
    # 1. warmup
    if warmup_epochs > 0:
        print('[INFO]  warmup 训练')
        model.fit_generator(train_data_gen,
                            epochs=warmup_epochs,
                            validation_data=valid_data_gen,
                            workers=train_workers)

    # 2. 正式训练
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


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] =  'TRUE'
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
