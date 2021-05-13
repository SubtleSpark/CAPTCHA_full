class Model:
    backend = "VGG"  # 使用的模型 NNModels 中的 模型名
    model_data = "./model_data"  # 训练生成的checkpoint和日志文件保存的文件夹
    model_path = "./model_data/cnn_best.h5"  # 加载的模型
    input_shape = (40, 120, 3)


class Train:
    """
    训练集
    """
    train_data_folder = "/root/work/captcha/A/train/"
    train_data_file = "/root/work/captcha/A/train/train_label.csv"
    train_prob_from = 0
    train_prob_to = 0.8

    """
    超参数
    """
    batch_size = 32
    learning_rate = 0.001
    warmup_epochs = 20
    nb_epochs = 200

    """
    模型数据及其他
    """
    workers = 4
    pretrained_weights = "./model_data/pre_weight.h5"  # 预训练权重
    saved_weights_name = "CNN_captcha_weight.h5"
    debug = True


class Valid:
    """
    验证集
    """
    valid_data_folder = "/root/work/captcha/A/train/"
    valid_data_file = "/root/work/captcha/A/train/train_label.csv"
    valid_prob_from = 0
    valid_prob_to = 0.8
    valid_times = 1


class Predict:
    predict_data_folder = "./data/test",
    predict_data_file = "./data/submission.csv"


"""
本地测试时需要重写的路径
"""
import sys

if sys.platform == "win32":
    Train.train_data_folder = "F:/data_set/captcha/A/train/"
    Train.train_data_file = "F:/data_set/captcha/A/train/train_label.csv"
    Train.batch_size = 2
    Train.warmup_epochs = 0
    Train.workers = 1

    Valid.valid_data_folder = "F:/data_set/captcha/A/train/"
    Valid.valid_data_file = "F:/data_set/captcha/A/train/train_label.csv"
