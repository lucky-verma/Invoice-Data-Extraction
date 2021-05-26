import os

import requests

from detect import transferFiles

runs = "runs/"
exp_path = os.path.join(runs, "detect")
best_model = "weights/model.pt"


def load_model():
    if best_model is None:
        url = "http://vast-ml-models.s3-ap-southeast-2.amazonaws.com/Invoice+All++Class+(Object+Detection)+best.pt"
        r = requests.get(url, allow_redirects=True)
        open('best.pt', 'wb').write(r.content)
        model = 'best.pt'
        if model is not None:
            print('Model DOWNLOADED from s3')
    else:
        model = best_model
    return model


# load_model()
jsonValue = transferFiles("test/test.png")
print(jsonValue)
