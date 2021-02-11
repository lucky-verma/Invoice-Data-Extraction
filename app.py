import streamlit as st
import time
import shutil
import os
import imutils
from pathlib import Path
from PIL import Image
import subprocess
import numpy as np
import wget

url = "https://awscdk-documentsbucket9ec9deb9-i5bemy0nz6wp.s3-ap-southeast-2.amazonaws.com/best.pt"
model = wget.download(url)
if model is not None:
    print('Model DOWNLOADED from s3')

runs = "runs/detect/"
exp_path = os.path.join(runs, "exp")




# @st.cache(allow_output_mutation=True)
# def load_model():
#     model = "yolov5/runs/train/exp/weights/best.pt"
#     return model


# python detect.py --weights runs/train/{ex(n)}/weights/best.pt --img 640 --conf 0.2 --source test/images

def run(model, conf, image):
    subprocess.run('ls', shell=True)
    subprocess.run('python detect.py --weights {model} --img 1024 --conf {conf} --source {image}'.format(model=model, image=image, conf=conf), shell=True)


# model = load_model()

st.title('VAST: Invoice Data Extraction')

file = st.file_uploader('Upload here', type=['jpg', 'png', 'jpeg', 'webp'])
st.write('## Adjust slider for precision Threshold')
slider = st.slider('Precision/Confidence Slider', min_value=0.4, max_value=0.9)

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
        subprocess.run('ls runs/detect/', shell=True)
        models = 'best.pt'
        run(models, slider, img)
        st.spinner()
        with st.spinner(text='In progress'):
            time.sleep(2)
            st.success('Done')
        st.balloons()
        print('modelling DONE')
        result_image = Image.open("runs/detect/exp/audacious.jpg")
        res_img_array = np.array(result_image)
        st.image(res_img_array, use_column_width=True)
        shutil.rmtree(exp_path, ignore_errors=False)
        try:
            if subprocess.run('ls runs/detect/', shell=True) is None:
                print('SUCCESSFULLY DELETED')
        except:
            pass
        subprocess.run('ls runs/detect/', shell=True)
        st.success('Success')
        pass
