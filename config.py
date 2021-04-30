class Model:
    backend = "tensorflow"
    model_data = "./model_data"
    model_path = "./model_data/cnn_best.h5"
    input_size = [128, 128]

    # def __str__(self) -> str:
    #     return "\n[INFO] model:" +\
    #            "\nbackend:\t" + self.backend +\
    #            "\nmodel_data:\t" + self.model_data +\
    #            "\nmodel_path:\t" + self.model_path +\
    #            "\ninput_size:\t" + self.input_size.__str__()


class Train:
    train_data_folder = "F:/data_set/captcha/A/train"
    train_data_file = "F:/data_set/captcha/A/train/train_label.csv"
    train_prob = [0, 0.8]
    pretrained_weights = "./model_data/pre_weight.h5"
    batch_size = 16
    learning_rate = 1e-3
    nb_epochs = 1
    warmup_epochs = 3
    saved_weights_name = "CNN_captcha_weight.h5"
    debug = True


class Valid:
    valid_data_folder = "F:/data_set/captcha/A/train",
    valid_data_file = "F:/data_set/captcha/A/train/train_label.csv",
    valid_prob = [0.8, 1],
    valid_times = 1


class Predict:
    predict_data_folder = "./data/test",
    predict_data_file = "./data/submission.csv"


