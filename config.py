class Model:
    backend = "tensorflow"
    model_data = "./model_data"
    model_path = "./model_data/cnn_best.h5"
    input_size = [128, 128]


class Train:
    """
    训练集
    """
    train_data_folder = "/root/work/captcha/A/train/"
    train_data_file = "/root/work/captcha/A/train/train_label.csv"
    train_prob = [0, 0.8]

    """
    超参数
    """
    batch_size = 64
    learning_rate = 0.001
    nb_epochs = 200
    warmup_epochs = 20

    """
    模型数据
    """
    workers = 4
    pretrained_weights = "./model_data/pre_weight.h5"
    saved_weights_name = "CNN_captcha_weight.h5"
    debug = True


class Valid:
    """
    验证集
    """
    valid_data_folder = "/root/work/captcha/A/train/"
    valid_data_file = "/root/work/captcha/A/train/train_label.csv"
    valid_prob = [0.8, 1],
    valid_times = 1


class Predict:
    predict_data_folder = "./data/test",
    predict_data_file = "./data/submission.csv"
