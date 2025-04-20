import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
from model import MainModel 
import numpy as np
from skimage.color import rgb2lab, lab2rgb


# Cache the model to avoid reloading it every time
@st.cache_resource
def load_model():
    model = MainModel()
    model.load_state_dict(torch.load("main_model_epoch_19.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# Load the pretrained model
model = load_model()
st.success("✅ Pretrained colorization model loaded")

# --- Upload Grayscale Image ---
uploaded_image = st.file_uploader("📷 Upload a grayscale image", type=["png", "jpg", "jpeg"])

if uploaded_image:
    # Convert to grayscale and display
    gray_image = Image.open(uploaded_image).convert("L")
    st.image(gray_image, caption="🖤 Grayscale Input", use_column_width=True)

    # --- Preprocess the Image ---
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.ToTensor(),         
        transforms.Normalize(mean=[0.5], std=[0.5])  
    ])
    input_tensor = transform(gray_image).unsqueeze(0) 

    # --- Colorization Inference ---
    if st.button("🎨 Colorize"):
        with torch.no_grad():
            # Predict ab channels using the model
            fake_ab = model.net_G(input_tensor)
            L = input_tensor[0][0].cpu().numpy() * 0.5 + 0.5  
            ab = fake_ab[0].cpu().numpy() * 0.5 + 0.5        

            lab = np.zeros((256, 256, 3))
            lab[:, :, 0] = L * 100                            
            lab[:, :, 1:] = (ab.transpose(1, 2, 0) - 0.5) * 256  
            rgb = lab2rgb(lab)
            rgb_image = (rgb * 255).astype(np.uint8)

            st.image(rgb_image, caption="🌈 Colorized Output", use_column_width=False, width=256)
