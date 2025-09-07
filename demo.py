import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
from model import MainModel
import numpy as np
from skimage.color import lab2rgb

# --------------------------
# Cache the model to avoid reloading every time
# --------------------------
@st.cache_resource
def load_model():
    model = MainModel()
    model.load_state_dict(torch.load("main_model_epoch_19.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# Load the pretrained model
model = load_model()
st.success("✅ Pretrained colorization model loaded")

# --------------------------
# Upload grayscale image
# --------------------------
uploaded_image = st.file_uploader("📷 Upload a grayscale image", type=["png", "jpg", "jpeg"])

if uploaded_image:
    # Convert to grayscale
    gray_image = Image.open(uploaded_image).convert("L")

    # Display input
    st.image(gray_image, caption="🖤 Grayscale Input", use_column_width=True)

    # --------------------------
    # Preprocess the image
    # --------------------------
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),         
        transforms.Normalize(mean=[0.5], std=[0.5])  # L channel ∈ [-1,1]
    ])
    input_tensor = transform(gray_image).unsqueeze(0)  # shape (1,1,256,256)

    # --------------------------
    # Colorization Inference
    # --------------------------
    if st.button("🎨 Colorize"):
        with torch.no_grad():
            fake_ab = model.net_G(input_tensor)  # predict ab channels

            # Denormalize L
            L = input_tensor[0][0].cpu().numpy()   # (256,256)
            L = (L + 1.0) * 50.0                  # back to [0,100]

            # Denormalize ab
            ab = fake_ab[0].cpu().numpy().transpose(1, 2, 0)  # (256,256,2)
            ab = ab * 110.0                                  # back to [-110,110]

            # Merge LAB
            lab = np.zeros((256, 256, 3))
            lab[:, :, 0] = np.clip(L, 0, 100)
            lab[:, :, 1:] = np.clip(ab, -128, 127)

            # Convert to RGB
            rgb = lab2rgb(lab)
            rgb_image = (rgb * 255).astype(np.uint8)

            # Display output
            st.image(rgb_image, caption="🌈 Colorized Output", use_column_width=True)
