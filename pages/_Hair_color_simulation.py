import streamlit as st
from streamlit_image_comparison import image_comparison
import cv2
import numpy as np
from pages.Hair_segmentation.Unet import UNet
import torch
from torchvision import transforms
from PIL import Image


# set page config
st.set_page_config(page_title="Hair color simulation", layout="centered")

#adding a file uploader
uploaded_file = st.file_uploader("Choose a picture you want to change hair color", type=['png', 'jpg'])

device = "cuda" if torch.cuda.is_available() else "cpu"
ratio = 0.3
alpha = 0.85
trns = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

if 'Unet' not in st.session_state:
    st.session_state['Unet'] = UNet(3, 1).to(device=device)
    checkpoint = torch.load('./pages/Hair_segmentation/model.pth', map_location=device)
    st.session_state['Unet'].load_state_dict(checkpoint['model_state_dict'])
    st.session_state['Unet'].eval()

face = cv2.cvtColor(cv2.imread("./pages/20.jpg"), cv2.COLOR_BGR2RGB)



if uploaded_file is not None:

    #To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    img_buffer_numpy = np.frombuffer(bytes_data, dtype=np.uint8)
    img = cv2.imdecode(img_buffer_numpy, 1)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img_tensor = trns(img).unsqueeze(0).to(device=device)
    logits_mask=st.session_state['Unet'](img_tensor.to('cuda'))
    pred_mask=torch.sigmoid(logits_mask)
    pred_mask=(pred_mask > ratio)*1.0
    mask = transforms.ToPILImage()((pred_mask.squeeze(0)))
    mask = cv2.resize(np.array(mask),(img.size[0], img.size[1]))
    color = st.color_picker('Choose your Color', '#4d0c83')
    color =  color.lstrip('#')
    red, green, blue = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
    _, mask = cv2.threshold(np.array(mask) , thresh=180, maxval=255, type=cv2.THRESH_BINARY)

    color_mask = np.copy(np.array(img))
    color_mask[mask==255] = [red, green, blue]

    
    color_hair = cv2.addWeighted(color_mask, 1-alpha, np.array(img), alpha, 0)


# render image-comparison

    image_comparison(
        img1=img,
        img2=color_hair,
    )








