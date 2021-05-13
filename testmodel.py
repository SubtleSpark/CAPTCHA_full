# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:32:59 2019

@author: qmzhang
"""

import json
import os

import numpy as np
import pandas as pd
from cv2 import resize, imread
from keras import Model
from keras.models import load_model

from util.imageProcess import img_procrss
from util.modelUtils import word_acc
from util import labelProcess


def model(testpath):
    # your model goes here
    # 在这里放入或者读入模型文件
    # model_path = r"./model_data/model.138-0.06.h5"  # 0.9774

    # load model
    model_path = r"./model_data/cnn_best.h5"
    model: Model = load_model(model_path, custom_objects={"my_acc": word_acc})

    # load data
    print("reading start!")
    pics_name = [str(x) + ".jpg" for x in range(1, 5001)]
    pics_path = [(testpath + pic_name) for pic_name in pics_name]
    X = load_data(pics_path=pics_path)
    print("reading end!")

    # predict
    predict = model.predict(X, batch_size=16)
    ans = labelProcess.decode_predict(predict)

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
    x = img_procrss(x)

    x = resize(x, dsize=(120, 40))
    return x


if __name__ == "__main__":
    testpath = './data/test/'  # 测试集路径。包含验证码图片文件的文件夹
    result_folder_path = './result.csv'  # 结果输出文件路径

    # 调用自己的工程文件，并这里生成结果文件(dataframe)
    result = model(testpath)
    print(result)

    # 注意路径不能更改，index需要设置为None
    result.to_csv(result_folder_path, index=None)
    # 参考代码结束：输出标准结果文件
