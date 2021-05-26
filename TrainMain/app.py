import streamlit as st
import time
import shutil
import os
import pandas as pd
import imutils
from pathlib import Path
from PIL import Image
import subprocess
import numpy as np
import wget
import requests
import urllib.request
from detect import transferFiles

runs = "runs/"
exp_path = os.path.join(runs, "detect")


@st.cache(allow_output_mutation=True)
def load_model():
    url = "http://vast-ml-models.s3-ap-southeast-2.amazonaws.com/Invoice+All++Class+(Object+Detection)+best.pt"
    r = requests.get(url, allow_redirects=True)
    open('model.pt', 'wb').write(r.content)
    model = 'model.pt'
    if model is not None:
        print('Model DOWNLOADED from s3')
    return model


# python detect.py --weights runs/train/{ex(n)}/weights/best.pt --img 640 --conf 0.2 --source test/images

load_model()

st.title('Invoice Data Extraction')
# st.write('## Adjust slider for precision Threshold')
# slider = st.slider('Precision/Confidence Slider', min_value=0.4, max_value=0.9)

file = st.file_uploader('Upload here', type=['jpg', 'png', 'jpeg', 'webp'])

if file is None:
    st.write("### Please upload your Invoice IMAGE")
else:
    im = Image.open(file)
    rgb_im = im.convert('RGB')
    rgb_im.save('audacious.jpg')
    os.getcwd()
    img = "audacious.jpg"
    st.image("audacious.jpg", caption='invoice?', use_column_width=True)
    if st.button("Process"):
        st.spinner()
        with st.spinner(text='In progress'):
            jsonValue = transferFiles(img)
        st.success('Done')
        st.balloons()
        print('modelling DONE')
        # for i in jsonValue.items():
        #     st.text(i)
        st.json(jsonValue)
        result_image = Image.open("runs/detect/exp/audacious.jpg")
        res_img_array = np.array(result_image)
        st.image(res_img_array, use_column_width=True)
        shutil.rmtree(exp_path, ignore_errors=False)
        subprocess.run('ls runs/detect/', shell=True)
        dir_name = os.getcwd()
        test = os.listdir(dir_name)
        # for item in test:
        #     if item.endswith(".pt"):
        #         os.remove(os.path.join(dir_name, item))
        #         print('dElEtInG')
        subprocess.run('ls', shell=True)
        st.success('Done')
        st.info('Thank You!')
        pass
