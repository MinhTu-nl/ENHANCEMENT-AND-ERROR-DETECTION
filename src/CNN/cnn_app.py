import streamlit as st
import torch
from model import Generator
from utils import save_image
import cv2
import numpy as np

st.title("Low Light Image Enhancement")

# Load the trained model
@st.cache_resource
def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator().to(device)  # Chuyển mô hình lên device
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# Load model with the latest checkpoint (e.g., epoch_200.pth)
checkpoint_path = "weights/CNN+GAN/generator_epoch_550.pth"
model = load_model(checkpoint_path)

# File uploader
uploaded_file = st.file_uploader("Choose a low-light image", type=["jpg", "png"])

if uploaded_file is not None:
    # Read and preprocess the image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target_size = (256, 256)  # Match the training image size
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Enhance Image"):
        # Convert image to tensor
        transform = lambda x: torch.from_numpy(x.transpose(2, 0, 1)).float() / 255.0
        input_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)  # Lấy device từ mô hình

        # Enhance image
        with torch.no_grad():
            enhanced_tensor = model(input_tensor)

        # Convert tensor back to image
        enhanced_image = enhanced_tensor[0].cpu().numpy().transpose(1, 2, 0) * 255.0
        enhanced_image = enhanced_image.astype(np.uint8)
        
        # Display enhanced image
        st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)

        # Optional: Save the enhanced image
        save_path = "CNN+GAN/enhanced_image.png"
        save_image(enhanced_tensor[0], save_path)
        st.write(f"Enhanced image saved as {save_path}")