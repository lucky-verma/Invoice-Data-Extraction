import base64
import io
import os

import requests
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel

from detect import transferFiles

app = FastAPI()
runs = "runs/"
exp_path = os.path.join(runs, "detect")
best_model = "weights/model.pt"


# def load_model():
#     if best_model is None:
#         url = "http://vast-ml-models.s3-ap-southeast-2.amazonaws.com/Invoice+All++Class+(Object+Detection)+best.pt"
#         r = requests.get(url, allow_redirects=True)
#         open('best.pt', 'wb').write(r.content)
#         model = 'best.pt'
#         if model is not None:
#             print('Model DOWNLOADED from s3')
#     else:
#         model = best_model
#     return model
#
#
# load_model()

model = best_model


# jsonValue = transferFiles("test/test.png")
# print(jsonValue)


# define the Input class
class Input(BaseModel):
    base64str: str
    threshold: float


def base64str_to_PILImage(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj)
    return img


@app.put("/predict")
def get_predictionbase64(d: Input):
    img = base64str_to_PILImage(d.base64str)
    img.save("temp.png")
    jsonValue = transferFiles("temp.png")
    return jsonValue
