import string

import cv2
import numpy as np
import config


def RGBAlgorithm(rgb_img, value=0.5, basedOnCurrentValue=True):
    """
      基于RGB空间亮度调整算法：
      主要是对RGB空间进行亮度调整。计算出调整系数后，调整手段主要有两种：
        1) 基于当前RGB值大小进行调整，即R、G、B值越大，调整的越大，
        例如：当前像素点为(100,200,50),调整系数1.1,则调整后为(110,220,55)；
        2) 不考虑RGB值大小的影响，即始终对各个点R、G、B值进行相同的调整，
        例如：当前像素点为(100,200,50),调整系数10/255,则调整后为(110,210,60)。
    """
    img = rgb_img * 1.0
    img_out = img

    # 基于当前RGB进行调整（RGB*alpha）
    if basedOnCurrentValue:
        # 增量大于0，指数调整
        if value >= 0:
            alpha = 1 - value
            alpha = 1 / alpha

        # 增量小于0，线性调整
        else:
            alpha = value + 1

        img_out[:, :, 0] = img[:, :, 0] * alpha
        img_out[:, :, 1] = img[:, :, 1] * alpha
        img_out[:, :, 2] = img[:, :, 2] * alpha

    # 独立于当前RGB进行调整（RGB+alpha*255）
    else:
        alpha = value
        img_out[:, :, 0] = img[:, :, 0] + 255.0 * alpha
        img_out[:, :, 1] = img[:, :, 1] + 255.0 * alpha
        img_out[:, :, 2] = img[:, :, 2] + 255.0 * alpha

    img_out = img_out / 255.0

    # RGB颜色上下限处理(小于0取0，大于1取1)
    mask_3 = img_out < 0
    mask_4 = img_out > 1
    img_out = img_out * (1 - mask_3)
    img_out = img_out * (1 - mask_4) + mask_4

    return img_out


def imgProcessNorm(img, shape):
    # 中值滤波
    res = cv2.medianBlur(img, ksize=3)
    # 亮度
    # res = RGBAlgorithm(res, value=0.5)

    # reshape
    res = cv2.resize(src=res, dsize=shape)
    # 归一化
    cv2.normalize(src=res, dst=res, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return res
