# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:32:59 2019

@author: qmzhang
"""

import json
import os

import pandas as pd
from cv2 import resize, imread
from keras import Model
from keras.models import load_model

from util import utils
from util.utils import *


def model(testpath):
    # your model goes here
    # 在这里放入或者读入模型文件
    # model_path = r"./model_data/model.138-0.06.h5"  # 0.9774

    # load model
    model_path = r"./model_data/cnn_best.h5"
    model: Model = load_model(model_path, custom_objects={"my_acc": my_acc})

    # load data
    print("reading start!")
    pics_name = [str(x) + ".jpg" for x in range(1, 5001)]
    pics_path = [(testpath + pic_name) for pic_name in pics_name]
    X = load_data(pics_path=pics_path)
    print("reading end!")

    # predict
    predict = model.predict(X, batch_size=16)
    ans = utils.decode_predict(predict)

    # the format of result-file
    # 这里可以生成结果文件
    ids = [str(x) + ".jpg" for x in range(1, 5001)]
    labels = ans
    df = pd.DataFrame([ids, labels]).T
    df.columns = ['ID', 'label']
    return df


def load_data(pics_path):
    data = []
    for path in pics_path:
        data.append(get_data(path))
    return np.array(data)


def get_data(path):
    # 取出一个x, 并resize
    x = imread(filename=path)

    # 去噪,并归一化
    x = utils.img_procrss(x)

    x = resize(x, dsize=(120, 40))
    return x
