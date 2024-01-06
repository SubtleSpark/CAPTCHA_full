from typing import List

from fastapi import FastAPI, UploadFile, File
import numpy as np
import config
import cv2
from keras import Model
from keras.models import load_model

from util.imageProcess import imgProcessNorm
from util.modelUtils import word_acc
from util import labelProcess

app = FastAPI()

model_path = r"./VGG.h5"
model: Model = load_model(model_path, custom_objects={"word_acc": word_acc})


@app.post("/captcha/recognition/")
async def upload_images(files: List[UploadFile] = File(...)):
    # 处理接收到的图片文件
    images = []
    for file in files:
        content = await file.read()
        image_array = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = imgProcessNorm(img=image, shape=config.Model.img_shape)
        images.append(image)

    predict = model.predict(np.array(images))
    pred = labelProcess.decode_predict(predict)
    print('预测结果：' + str(pred))
    return {"message": "Images uploaded successfully", "images": pred}
