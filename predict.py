import json
import os
import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from cv2 import resize, imread
from keras import Model
from keras.models import load_model

import config
from util.imageProcess import imgProcessNorm
from util.modelUtils import word_acc
from util import labelProcess


def model(model_path, testpath):
    # your model goes here
    # 在这里放入或者读入模型文件
    # model_path = r"./model_data/model.138-0.06.h5"  # 0.9774

    # load model
    model: Model = load_model(model_path, custom_objects={"word_acc": word_acc})

    # load data
    print("[INFO] reading start!")
    pics_name = [str(x) + ".jpg" for x in range(1, 5001)]
    pics_path = [(testpath + pic_name) for pic_name in pics_name]
    X = load_data(pics_path=pics_path)
    print("[INFO] reading end!")

    # predict
    statr = time.time()
    print("[INFO] predict start! " + str(statr))
    predict = model.predict(X, batch_size=16)
    ans = labelProcess.decode_predict(predict)
    end = time.time()
    print("[INFO] predict end! " + str(end))
    print(end - statr)

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
    x = imgProcessNorm(x, config.Model.img_shape)
    return x


if __name__ == "__main__":
    testpath = config.Predict.predict_data_folder  # 测试集路径。包含验证码图片文件的文件夹
    result_folder_path = config.Predict.predict_result_file  # 结果输出文件路径
    model_path = config.Model.model_path

    # 参数
    parser = ArgumentParser()
    parser.add_argument('-m', '--model')
    args = parser.parse_args()

    if args.model is not None:
        model_path = args.model
        print('[INFO] ArgumentParser: model = ' + args.model)

    result = model(model_path, testpath)
    result.to_csv(result_folder_path, index=None)
