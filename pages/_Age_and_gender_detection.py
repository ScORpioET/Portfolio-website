#importing required libraries

import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np
import cv2
import torch
import sys
from torchvision import transforms
import os
from pages.Face_recognition_and_detection.MTL import MTL
from PIL import Image

st.write("# Age and gender detection")

st.write('''Here is face age and gender detection for Asians.\tA total of two models were used in this project.\n
GihHub：https://github.com/ScORpioET/Face-recognition-and-detection
''')

st.write('''### Face recognition\n
Model：Resnet50\n
Dataset：AFAD\n
Trainloss\n
''')
st.image('./pages/Face_recognition_and_detection/img/face_recognition_train_loss1-472.png')

st.write('''### Face recognition\n
Model：Yolov5s\n
Dataset：Celeba\n
Trainloss(1-7epochs)\n
''')
st.image('./pages/Face_recognition_and_detection/img/face_detection_loss1-7.png')
st.write('Trainloss(8-10epochs)')
st.image('./pages/Face_recognition_and_detection/img/face_detection_loss8-10.png')

st.write('## You can demo it below')

#adding a file uploader
uploaded_file = st.file_uploader("Choose the face you want to predict", type=['png', 'jpg'])

@st.cache_data
def load_model():
    Recognition_model = MTL()
    Recognition_model.load_state_dict(torch.load('./pages/Face_recognition_and_detection/FaceRecognition_model.pth', map_location=torch.device('cpu')))
    Recognition_model.eval()
    Detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./pages/Face_recognition_and_detection/FaceDetection_model.pt')
    return Recognition_model, Detection_model


Recognition_model, Detection_model = load_model()


if uploaded_file is not None:

    #To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    img_buffer_numpy = np.frombuffer(bytes_data, dtype=np.uint8)
    img = cv2.imdecode(img_buffer_numpy, 1)
    
    st.image(img, channels='BGR')

    button = st.button('Predict')

    if button:
        Bboxs = Detection_model(img)

        if(len(Bboxs.pandas().xyxy[0]["name"].values)==0):
            st.error('### no people detected', icon="🚨")
            st.info('This is a purely informational message', icon="ℹ️")
        else:
            for _, Bbox in Bboxs.pandas().xyxy[0].iterrows():
                bbox_xmin = int(Bbox["xmin"])
                bbox_xmax = int(Bbox["xmax"])
                bbox_ymin = int(Bbox["ymin"])
                bbox_ymax = int(Bbox["ymax"])

                capture = Image.fromarray(cv2.cvtColor(img[bbox_ymin:bbox_ymax,bbox_xmin:bbox_xmax], cv2.COLOR_BGR2RGB))

                trns = transforms.Compose([
                                transforms.Resize((32, 32)), 
                                transforms.ToTensor(), 
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                img_tensor = trns(capture).unsqueeze(0)

                age_output, gender_output = Recognition_model(img_tensor)

                age = round(age_output.item())
                gender_list = gender_output.tolist()[0]
                gender = 'male' if gender_list[0] >= gender_list[1] else 'female'

                # cv2.rectangle(img, (bbox_xmin,bbox_ymin), (bbox_xmax,bbox_ymax), (0,255,0), 6)
                
                st.image(img[bbox_ymin:bbox_ymax,bbox_xmin:bbox_xmax], channels='BGR')
                st.info(f'### age：{age}\n ### gender：{gender}', icon="ℹ️")
