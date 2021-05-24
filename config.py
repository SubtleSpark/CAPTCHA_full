class Model:
    """
    模型相关参数
    """
    model = "VGG"  # 使用的模型 NNModels 中的 模型名
    model_data = "./model_data"  # 训练生成的checkpoint和日志文件保存的文件夹
    model_path = None  # 加载的模型
    __h__ = 64  # 图片高度
    __w__ = 256  # 图片宽度
    input_shape = (__h__, __w__, 3)  # 模型输入为（h, w, c）  cv的图像大小为(w, h)
    img_shape = (__w__, __h__)


class Train:
    """
    训练集
    """
    train_data_folder = "/root/work/captcha/train/"
    train_data_file = "/root/work/captcha/train/train_label.csv"
    train_prob_from = 0
    train_prob_to = 1

    """
    超参数
    """
    batch_size = 64
    learning_rate = 0.001
    warmup_epochs = 0
    nb_epochs = 500

    """
    模型数据及其他
    """
    workers = 4
    use_preweight = False
    pretrained_weights = "/root/work/CAPTCHA_full/model_data/KerasResNet50/checkpoints/KerasResNet50.h5"  # 预训练权重
    saved_weights_name = "CNN_captcha_weight.h5"
    debug = True


class Valid:
    """
    验证集
    """
    valid_data_folder = "/root/work/captcha/test/"
    valid_data_file = "/root/work/captcha/test/test_label.csv"
    valid_prob_from = 0
    valid_prob_to = 1
    valid_times = 1


class Predict:
    predict_data_folder = "./data/test",
    predict_data_file = "./data/submission.csv"


class Merge:
    model_paths = [r'/root/work/model_data/KerasResNet50/checkpoints/KerasResNet50.h5',
                   r'/root/work/model_data/VGG/checkpoints/VGG.h5',
                   r'/root/work/model_data/SEResNet50/checkpoints/SEResNet50.h5'
                   ]


"""
本地测试时需要重写的路径
"""
import sys

if sys.platform == "win32":
    Train.train_data_folder = "F:/data_set/captcha/train/"
    Train.train_data_file = "F:/data_set/captcha/train/train_label.csv"
    Train.train_prob_from = 0
    Train.train_prob_to = 0.1
    Train.batch_size = 2
    Train.warmup_epochs = 0
    Train.workers = 1

    Valid.valid_data_folder = "F:/data_set/captcha/train/"
    Valid.valid_data_file = "F:/data_set/captcha/train/train_label.csv"
    Valid.valid_prob_from = 0.1
    Valid.valid_prob_to = 0.2

    Merge.model_paths = [r'F:\model_data\KerasResNet50\checkpoints\KerasResNet50.h5',
                         r'F:\model_data\KerasResNet50\checkpoints\KerasResNet50.h5'
                         ]
