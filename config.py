class Model:
    """
    模型相关参数
    """
    model = "VGG"  # 使用的模型 NNModels 中的 模型名
    model_data = "./model_data"  # 训练生成的checkpoint和日志文件保存的文件夹
    model_path = "/root/work/model_data/KerasResNet50/checkpoints/KerasResNet50.h5"  # 加载的模型
    __h__ = 32  # 图片高度
    __w__ = 128  # 图片宽度
    input_shape = (__h__, __w__, 3)  # 模型输入为（h, w, c）  cv的图像大小为(w, h)
    img_shape = (__w__, __h__)


class Train:
    """
    训练集
    """
    train_data_folder = "E:\yzm"
    train_prob_from = 0
    train_prob_to = 0.9

    """
    超参数
    """
    batch_size = 32
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
    valid_data_folder = "E:\yzm"
    valid_prob_from = 0.9
    valid_prob_to = 1
    valid_times = 1


class Predict:
    predict_data_folder = "/root/work/captcha/test/"
    predict_result_file = "./result.csv"


class Merge:
    model_paths = [r'/root/work/model_data/KerasResNet50/checkpoints/KerasResNet50.h5',
                   # r'/root/work/model_data/VGG/checkpoints/VGG.h5',
                   r'/root/work/model_data/SEResNet50/checkpoints/SEResNet50.h5'
                   ]

