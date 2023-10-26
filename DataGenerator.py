import os

import numpy as np
from cv2 import imread  # BGR
from keras.utils import Sequence

from util import imageProcess, imageArgument, labelProcess


class DataGenerator(Sequence):
    '''
    :parameter
        data_file: 保存数据集的文本文件路径
        data_dir: 保存数据集文件夹
        prob:       list 用于划分训练集
        img_shape=(w, h)
    '''

    def __init__(self, data_dir, img_shape=(128, 128), batch_size=16, data_aug=False, prob_from=0, prob_to=1,
                 shuffle=True):
        self.data_dir = data_dir
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.data_aug = data_aug
        self.shuffle = shuffle

        # 数据增强器
        self.aug_pipe = imageArgument.getAugPipe()

        # 数据字典
        self.filename_list = os.listdir(data_dir)
        self.data_map = {}

        # 划分数据
        begin = int(len(self.filename_list) * prob_from)
        end = int(len(self.filename_list) * prob_to)
        self.filename_list = self.filename_list[begin:end]

        # 从data_list获取全部X, Y
        for line in self.filename_list:
            data = self.get_data(line)
            self.data_map.update({line: data})

        # 随机打乱data_list
        if self.shuffle:
            np.random.shuffle(self.filename_list)

    def __len__(self):
        return int(np.ceil(len(self.filename_list) / float(self.batch_size)))

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle:
            np.random.shuffle(self.filename_list)

    def __getitem__(self, idx):
        batch = self.filename_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        for line in batch:
            # 读取出x, y
            [x, y] = self.data_map[line]

            # 数据增强
            if self.data_aug:
                x = self.aug_pipe.augment_image(x)

            batch_x.append(x)
            batch_y.append(y)

        return np.array(batch_x), np.array(batch_y)

    # 取出X， Y
    def get_data(self, filename: str):
        # 根据文件名，获取标签图片名、标签
        label = filename[0:4]

        # 取出一个图片x
        path = os.path.join(self.data_dir, filename)
        x = imread(filename=path)

        # 预处理并归一化
        x = imageProcess.imgProcessNorm(x, self.img_shape)

        y = labelProcess.process_label(label)
        return x, y


if __name__ == '__main__':
    gen = DataGenerator(r'E:\yzm', (128, 32), 16)
    pair = gen.__getitem__(1)
    print(pair)