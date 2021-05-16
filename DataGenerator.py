import os
import numpy as np
import cv2

from cv2 import resize, imread, imshow, waitKey  # BGR
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

    def __init__(self, data_file, data_dir, img_shape=(128, 128), batch_size=16, data_aug=False, prob_from=0, prob_to=1,
                 shuffle=True):
        self.data_dir = data_dir
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.data_aug = data_aug
        self.shuffle = shuffle

        # 数据增强器
        self.aug_pipe = imageArgument.getAugPipe()

        # 数据字典
        self.data_list = []
        self.data_map = {}

        # 从文件中读取data_list
        with open(data_file, 'r') as file:
            for line in file.readlines():
                curLine = line.strip()
                if curLine == '':
                    continue
                self.data_list.append(curLine)
        del self.data_list[0]  # 删除标题行

        # 划分数据
        begin = int(len(self.data_list) * prob_from)
        end = int(len(self.data_list) * prob_to)
        self.data_list = self.data_list[begin:end]

        # 从data_list获取全部X, Y
        for line in self.data_list:
            data = self.get_data(line)
            self.data_map.update({line: data})

        # 随机打乱data_list
        if self.shuffle:
            np.random.shuffle(self.data_list)

    def __len__(self):
        return int(np.ceil(len(self.data_list) / float(self.batch_size)))

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle:
            np.random.shuffle(self.data_list)

    def __getitem__(self, idx):
        batch = self.data_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        for line in batch:
            # 读取出x, y
            [x, y] = self.data_map[line]
            batch_x.append(x)
            batch_y.append(y)

        return np.array(batch_x), np.array(batch_y)

    # 取出X， Y
    def get_data(self, line):
        # 根据文本文件的一行，获得图片名、标签
        imgname, label = line.split(',')

        # 取出一个图片x
        path = os.path.join(self.data_dir, imgname)
        x = imread(filename=path)

        # 数据增强
        if self.data_aug:
            x = self.aug_pipe.augment_image(x)

        # 预处理并归一化
        x = imageProcess.imgProcessNorm(x, self.img_shape)

        y = labelProcess.process_label(label)
        return x, y
