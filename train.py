import os
import config
from util.modelUtils import word_acc
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.optimizers import *
from DataGenerator import DataGenerator
import NNModels
from NNModels import LeNet, SEResNet50, KerasResNet50

"""
1. 读取配置文件
2. 加载各个类型的配置数据
"""
# model
model_data = config.Model.model_data

# train
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


def main(weight_path: str = None):
    # 加载模型结构
    nnm = SEResNet50(inputShape=((40, 120, 3)), regularizer=0.001, droprate=0.5)
    model = nnm.getModel()

    """
    根据参数加载模型数据
    """
    # 读取权重
    if weight_path != None:
        try:
            print(weight_path)
            model.load_weights(filepath=pretrained_weights, by_name=True, skip_mismatch=True)
        except:
            print('pre weights not load')
        else:
            print('pre weights has load')

    plot_model(model=model, to_file=os.path.join(model_data, 'model.png'), show_shapes=True)
    # model.summary()

    # 编译模型
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.001, amsgrad=True),
                  metrics=['accuracy', word_acc])

    """
    创建数据生成器
    """
    # 训练数据生成器
    train_data_gen = DataGenerator(data_file=train_file, data_dir=train_dir, img_shape=(120, 40), batch_size=batch_size,
                                   data_aug=True,
                                   prob_from=train_prob_from,
                                   prob_to=train_prob_to,
                                   shuffle=True)
    # 验证数据生成器
    valid_data_gen = DataGenerator(data_file=valid_file, data_dir=valid_dir, img_shape=(120, 40), batch_size=batch_size,
                                   data_aug=False,
                                   prob_from=valid_prob_from,
                                   prob_to=valid_prob_to,
                                   shuffle=True)

    """
    训练：
    1. warmup
    2. 正式训练
    """
    # 1. warmup
    print(model.optimizer)
    model.fit_generator(train_data_gen,
                        epochs=warmup_epochs,
                        validation_data=valid_data_gen,
                        workers=train_workers)
    # 2. 正式训练
    callbacks = [  # EarlyStopping(monitor='val_loss', patience=10),
        CSVLogger(os.path.join(model_data, 'train_log.csv')),

        # filepath = "./model_data/model.{epoch:02d}-{val_loss:.8f}.h5"
        ModelCheckpoint(filepath="./model_data/model.{epoch:02d}-{val_loss:.8f}.h5",
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=False,
                        save_weights_only=False,
                        mode='auto',
                        period=2)]

    model.fit_generator(train_data_gen,
                        epochs=200,
                        validation_data=valid_data_gen,
                        workers=train_workers,
                        use_multiprocessing=True,
                        callbacks=callbacks)


if __name__ == '__main__':
    main()
