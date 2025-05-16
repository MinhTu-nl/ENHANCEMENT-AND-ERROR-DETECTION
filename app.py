import streamlit as st
import torch
from src.CNN.model import Generator
from src.CNN.utils import save_image
import streamlit as st
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import uuid
import pandas as pd
from datetime import datetime

# Thiết lập cấu hình trang
st.set_page_config(page_title="PROJECT AI", layout="wide")
st.title("Ứng Dụng AI Để Phát Hiện Lỗi Trong Môi Trường Ánh Sáng Yếu")

# Khởi tạo session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = {}

# --- Tải mô hình CNN ---
@st.cache_resource
def load_cnn_model(checkpoint_path):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Generator().to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình CNN: {e}")
        return None, None

cnn_checkpoint_path = "weights/CNN+GAN/generator_epoch_550.pth"
cnn_model, device = load_cnn_model(cnn_checkpoint_path)

# --- Tải mô hình YOLO ---
@st.cache_resource
def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình YOLO: {e}")
        return None

# Định nghĩa các mô hình YOLO và tên lớp tương ứng
product_models = {
    "Cáp": {
        "path": "weights/Yolo11n/CABLE_EXP/weights/best.pt",
        "classes": {
            0: "Dây bị uốn cong",
            1: "Dây bị hoán đổi",
            2: "Kết hợp",
            3: "Cắt lớp cách điện bên trong",
            4: "Cắt lớp cách điện bên ngoài",
            5: "Thiếu dây",
            6: "Thiếu cáp",
            7: "Đâm thủng lớp cách điện"
        }
    },
    "Chai lọ": {
        "path": "weights/Yolo11n/BOTTLE_EXP/weights/best.pt",
        "classes": {
            0: "Vỡ lớn",
            1: "Vỡ nhỏ",
            2: "Nhiễm bẩn"
        }
    },
    "Ốc Vít": {
        "path": "weights/Yolo11n/SCREW_EXP/weights/best.pt",
        "classes": {
            0: "Mặt trước bị thao tác",
            1: "Đầu bị xước",
            2: "Cổ bị xước",
            3: "Ren bên bị lỗi",
            4: "Ren trên bị lỗi"
        }
    },
    "Mạch bán dẫn": {
        "path": "weights/Yolo11n/TRANSISTOR_EXP/weights/best.pt",
        "classes": {
            0: "Chân bị uốn cong",
            1: "Chân bị cắt",
            2: "Vỏ bị hỏng",
            3: "Đặt sai vị trí"
        }
    },
    "Khóa kéo": {
        "path": "weights/Yolo11n/ZIPPER_EXP/weights/best.pt",
        "classes": {
            0: "Răng bị gãy",
            1: "Kết hợp",
            2: "Vải ở biên bị lỗi",
            3: "Vải bên trong bị lỗi",
            4: "Thô ráp",
            5: "Răng bị tách",
            6: "Răng bị ép"
        }
    }
}

# --- Sidebar ---
with st.sidebar:
    st.header("Cài đặt")
    st.subheader("Chọn sản phẩm")
    selected_product = st.radio(
        "Chọn một sản phẩm để phát hiện lỗi:",
        list(product_models.keys()),
        index=0
    )
    confidence_threshold = st.slider(
        "Ngưỡng tin cậy (Confidence Threshold)", 0.0, 1.0, 0.35, 0.05
    )
    enhance_option = st.checkbox("Bật chế độ tăng cường ánh sáng", value=True)
    img_size = 640

# Tải mô hình YOLO tương ứng
yolo_model = load_yolo_model(product_models[selected_product]["path"])
class_names = product_models[selected_product]["classes"]

# --- Xử lý ảnh ---
def process_image(image_file, cnn_model, yolo_model, device, confidence, selected_product, enhance_option):
    results = {}
    try:
        # Đọc ảnh
        image = Image.open(image_file).convert('RGB')
        results['original'] = image
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_np = cv2.resize(image_np, (256, 256), interpolation=cv2.INTER_AREA)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Tăng cường ảnh nếu được bật
        if enhance_option and cnn_model is not None:
            transform = lambda x: torch.from_numpy(x.transpose(2, 0, 1)).float() / 255.0
            input_tensor = transform(image_np).unsqueeze(0).to(device)
            with torch.no_grad():
                enhanced_tensor = cnn_model(input_tensor)
            enhanced_image = enhanced_tensor[0].cpu().numpy().transpose(1, 2, 0) * 255.0
            enhanced_image = enhanced_image.astype(np.uint8)
            results['enhanced'] = enhanced_image
        else:
            results['enhanced'] = None
        
        # Phát hiện lỗi
        if yolo_model is not None:
            yolo_input = cv2.cvtColor(enhanced_image if results['enhanced'] is not None else image_np, cv2.COLOR_RGB2BGR)
            yolo_results = yolo_model.predict(
                source=yolo_input,
                conf=confidence,
                imgsz=img_size
            )
            annotated_image_bgr = yolo_results[0].plot()
            annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
            results['detected'] = annotated_image_rgb
            results['detected_bgr'] = annotated_image_bgr
            
            # Chi tiết lỗi
            detected_objects = yolo_results[0].boxes.data.cpu().numpy()
            detection_details = []
            if len(detected_objects) > 0:
                for i, det in enumerate(detected_objects):
                    x1, y1, x2, y2, conf, cls_id = det
                    class_name = class_names.get(int(cls_id), "Không xác định")
                    detection_details.append(
                        f"- **{class_name}** (Độ tin cậy: {conf:.2f}) - Vị trí: [x1={int(x1)}, y1={int(y1)}, x2={int(x2)}, y2={int(y2)}]"
                    )
            else:
                detection_details.append("✅ Không phát hiện lỗi nào với ngưỡng tin cậy hiện tại!")
            results['details'] = detection_details
        else:
            results['detected'] = None
            results['details'] = ["Không thể phát hiện lỗi do lỗi mô hình YOLO."]
        
        return results
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi xử lý ảnh: {e}")
        return None

# --- Tải ảnh lên ---
st.write("Tải lên ảnh sản phẩm tối (JPG, PNG, JPEG) - Giới hạn 200MB")
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], key="file_uploader")

# Xử lý ảnh mới
if uploaded_file is not None and uploaded_file != st.session_state.get('last_uploaded_file', None):
    try:
        st.session_state.current_image = uploaded_file
        st.session_state.last_uploaded_file = uploaded_file
    except Exception as e:
        st.error(f"Lỗi khi đọc ảnh: {e}")

# Hiển thị ảnh gốc ngay khi tải lên
if st.session_state.current_image is not None:
    try:
        image = Image.open(st.session_state.current_image).convert('RGB')
        st.subheader("Ảnh gốc")
        st.image(image, use_container_width=True)
        
        # Nút xử lý
        if st.button("Bắt đầu xử lý", type="primary"):
            with st.spinner("Đang xử lý..."):
                results = process_image(
                    st.session_state.current_image,
                    cnn_model,
                    yolo_model,
                    device,
                    confidence_threshold,
                    selected_product,
                    enhance_option
                )
                
                if results:
                    st.session_state.processed_results = results
                    
                    # Hiển thị ảnh (3 ảnh hàng ngang)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.subheader("Ảnh gốc")
                        st.image(results['original'], use_container_width=True)
                    with col2:
                        st.subheader("Ảnh tăng cường")
                        if results.get('enhanced') is not None:
                            st.image(results['enhanced'], use_container_width=True)
                        else:
                            st.write("Tăng cường ánh sáng bị tắt")
                    with col3:
                        if results.get('detected') is not None:
                            st.subheader("Kết quả phát hiện")
                            st.image(results['detected'], use_container_width=True)
                    
                    # Chi tiết lỗi
                    st.subheader(f"Chi tiết lỗi sản phẩm: {selected_product}")
                    if results['details']:
                        st.markdown("\n".join(results['details']))
                    else:
                        st.success("✅ Không phát hiện lỗi nào với ngưỡng tin cậy hiện tại!")
                    
                    # Lưu vào lịch sử
                    history_entry = {
                        "id": str(uuid.uuid4()),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "product": selected_product,
                        "details": "; ".join([d.replace("✅ ", "") for d in results['details']]),
                        "original_image": results['original'],
                        "enhanced_image": results['enhanced'],
                        "detected_image": results['detected']
                    }
                    st.session_state.history.append(history_entry)
                
    except Exception as e:
        st.error(f"Lỗi khi hiển thị ảnh: {e}")

# --- Nút xóa và tải ảnh ---
st.markdown("---")
col_clear, col_download_enhanced, col_download_detected = st.columns(3)
with col_clear:
    if st.button("Xóa ảnh"):
        st.session_state.current_image = None
        st.session_state.last_uploaded_file = None
        st.session_state.processed_results = {}
        st.rerun()
with col_download_enhanced:
    if st.session_state.processed_results.get('enhanced') is not None:
        _, img_buffer = cv2.imencode(".png", cv2.cvtColor(st.session_state.processed_results['enhanced'], cv2.COLOR_RGB2BGR))
        st.download_button(
            label="Tải ảnh tăng cường",
            data=img_buffer.tobytes(),
            file_name="enhanced_image.png",
            mime="image/png"
        )
with col_download_detected:
    if st.session_state.processed_results.get('detected_bgr') is not None:
        _, img_buffer = cv2.imencode(".png", st.session_state.processed_results['detected_bgr'])
        st.download_button(
            label="Tải ảnh phát hiện",
            data=img_buffer.tobytes(),
            file_name="detected_image.png",
            mime="image/png"
        )

# --- Lịch sử ---
st.markdown("---")
st.subheader("Lịch sử xử lý")
if st.session_state.history:
    for i, item in enumerate(st.session_state.history):
        with st.expander(f"Kết quả xử lý #{i+1} - {item['timestamp']} - {item['product']}", expanded=False):
            cols = st.columns(3)
            with cols[0]:
                st.image(item['original_image'], caption="Ảnh gốc", use_container_width=True)
            with cols[1]:
                if item['enhanced_image'] is not None:
                    st.image(item['enhanced_image'], caption="Ảnh tăng cường", use_container_width=True)
                else:
                    st.write("Tăng cường ánh sáng bị tắt")
            with cols[2]:
                if item['detected_image'] is not None:
                    st.image(item['detected_image'], caption="Kết quả phát hiện", use_container_width=True)
            
            st.markdown(f"**Chi tiết lỗi trên {item['product']}:**")
            if item['details'] and "Không phát hiện" not in item['details']:
                st.markdown(item['details'].replace("; ", "\n"))
            else:
                st.success("✅ Không phát hiện lỗi nào với ngưỡng tin cậy hiện tại!")
else:
    st.write("Chưa có lịch sử xử lý.")

st.markdown("---")
st.caption("@2025 Copyright by Nguyen Le Minh Tu")