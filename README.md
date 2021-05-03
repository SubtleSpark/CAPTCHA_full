# CAPTCHA_full 验证码识别完整项目

## 1. 在config.json中配置数据集路径

（以后可能将config.py作为配置文件）

`config.json` 中的属性 示例及解释
```json
{
  "model": {
    "backend": "tensorflow",
    "model_data": "./model_data",             // train后保存的目录
    "model_path": "./model_data/cnn_best.h5", // 预测时读取的模型文件
    "input_size": [         // 神经网络输入大小
      128,
      128
    ]
  },
  
  "train": {      // 训练配置
    "train_data_folder": "F:/data_set/captcha/A/train",     // 存放图片的文件夹：通过与图片名拼接，形成完整路径
    "train_data_file": "F:/data_set/captcha/A/train/train_label.csv",   // 一个csv文件：存放<图片名， 正确标签>
    "train_prob": [     // 将 train_data_file 文件的 0% ~ 80%的部分作为训练集
      0,
      0.8
    ],
    "pretrained_weights": "./model_data/pre_weight.h5",   // 加载这个模型文件进行训练
    "batch_size": 16, 
    "learning_rate": 1e-3,      // 学习率
    "nb_epochs": 1,
    "warmup_epochs": 3,
    "saved_weights_name": "CNN_captcha_weight.h5",        // 训练后以该文件名保存
    "debug": true
  },
  
  "valid": {    // 验证配置
    "valid_data_folder": "F:/data_set/captcha/A/train",   // 存放图片的文件夹：通过与图片名拼接，形成完整路径
    "valid_data_file": "F:/data_set/captcha/A/train/train_label.csv",   // 一个csv文件：存放<图片名， 正确标签>
    "valid_prob": [   // 将 valid_data_file 文件的 80% ~ 100%的部分作为训练集
      0.8,
      1
    ],
    "valid_times": 1
  },
  
  "predict": {
    "predict_data_folder": "./data/test",         // 要预测的图片文件夹
    "predict_data_file": "./data/submission.csv"  // 预测结果输出到submission.csv文件中
  }
}
```

## 2.运行`train.py`训练模型
1. 运行`train.py`训练模型

2. `model_data/train_log.csv`会实时记录训练数据

3. 模型会保存在`model_data`文件夹下

## 3.


## 参考思路
![](参考思路.jpg)