import string

import cv2
import numpy as np
import keras.backend as K
from keras.metrics import categorical_accuracy

wordlist = list('0123456789') + list(string.ascii_letters)


# 处理标签
def process_label(label):
    pro_lab = np.zeros(shape=(4, 62))

    label = list(label)
    for i in range(4):
        c = label[i]
        pro_lab[i][wordlist.index(c)] = 1
    return pro_lab


def decode(arr):
    """
    :param arr: 一个shape为(4， 64)的二维数组
    :return:    该数组对应的字符串   len = 4
    """
    arr = arr.tolist()
    result = ''

    idx = list(np.argmax(arr, axis=1))

    for i in idx:
        result += wordlist[i]

    return result


def decode_predict(predict):
    """
    :param predict: 一个list(ndarray) 其中ndarray.shape = (4, 64), 或者一个shape = (None, 4， 64)的三维数组
    :return:        解码后的字符串列表 list(str) str.len = 4
    """
    ans = []
    for arr in predict:
        ans.append(decode(arr))
    return ans


if __name__ == '__main__':
    """
    测试标签处理功能
    """
    lab1 = process_label('0aa0')
    lab2 = np.zeros(shape=(4, 62))
    lab2[0][10] = 0.9
    lab2[1][2] = 0.9
    lab2[2][2] = 0.9
    lab2[3][2] = 0.9
    res = decode_predict([lab1, lab2])
    print(res)
    print(lab1)