import streamlit as st
import shutil
import os
import zipfile

if not os.path.isfile("./pages/Face_recognition_and_detection/FaceRecognition_model.pth"):
    try:   
        os.system("type FaceRecognition_model.zip.001 FaceRecognition_model.zip.002 > ./FaceRecognition_model.zip")
        os.remove('FaceRecognition_model.zip.002')
        os.remove('FaceRecognition_model.zip.001')
    except:
        pass
    with zipfile.ZipFile('FaceRecognition_model.zip', 'r') as zip_ref:
        zip_ref.extractall('./pages/Face_recognition_and_detection')
    # shutil.move("./FaceRecognition_model.pth", "./pages/Face_recognition_and_detection/FaceRecognition_model.pth")

if not os.path.isfile("./pages/Hair_segmentation/model.pth"):
    try:
        os.system("type model.zip.001 model.zip.002 model.zip.003 model.zip.004 > model.zip")
        os.remove('model.zip.001')
        os.remove('model.zip.002')
        os.remove('model.zip.003')
        os.remove('model.zip.004')
    except:
        pass
    with zipfile.ZipFile('model.zip', 'r') as zip_ref:
        zip_ref.extractall('./pages/Hair_segmentation/')

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to my portfolio! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    This is a place to showcase my Machine Learning and Data Science projects.
    **👈 Select the sidebar** to see some projects.
"""
)

st.markdown("![Alt Text](https://64.media.tumblr.com/1b2e59310a622150071a7b2513f530fe/9d06023dd253e49e-eb/s640x960/94190608daa3f1fff3477f9b70745a2e25bfa124.gifv)")