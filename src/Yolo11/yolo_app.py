import streamlit as st
st.set_page_config(page_title="Phát Hiện Lỗi Chai Lọ", layout="wide")
st.title("Ứng Dụng Phát Hiện Lỗi Sản Phẩm Chai Lọ (YOLOv5)")
st.write("Tải lên hình ảnh chai lọ để phát hiện các khuyết tật như vỡ hoặc nhiễm bẩn.")
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2 # Sử dụng OpenCV để xử lý ảnh nếu cần
import os

# --- Cấu hình ---
MODEL_PATH = 'weights/Yolo11n/ZIPPER_EXP/weights/best.pt' # Đường dẫn đến file best.pt của bạn
CONFIDENCE_THRESHOLD = 0.35 
IMG_SIZE = 640

# --- Tải mô hình ---
# Sử dụng cache của Streamlit để chỉ tải mô hình một lần
@st.cache_resource
def load_yolo_model(model_path):
    """Tải mô hình YOLO từ đường dẫn."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        return None

model = load_yolo_model(MODEL_PATH)

# --- Giao diện Streamlit ---

# Sidebar cho cài đặt
with st.sidebar:
    st.header("Cài đặt")
    confidence_slider = st.slider(
        "Ngưỡng tin cậy (Confidence Threshold)", 0.0, 1.0, CONFIDENCE_THRESHOLD, 0.05
    )

uploaded_file = st.file_uploader("Chọn một file ảnh", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB') # Đảm bảo ảnh ở định dạng RGB
        image_np = np.array(image) # Chuyển PIL Image thành numpy array (RGB)

        st.image(image, caption='Ảnh đã tải lên.', use_container_width=True)
        st.write("")

        if st.button('Bắt đầu Phát hiện Lỗi'):
            if model is not None:
                with st.spinner('Đang xử lý...'):
                    # Chạy dự đoán
                    results = model.predict(
                        source=image_np, # Truyền ảnh numpy array
                        conf=confidence_slider,
                        imgsz=IMG_SIZE 
                    )

                    annotated_image_bgr = results[0].plot()
                    annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB) # Chuyển sang RGB để hiển thị đúng màu

                    st.image(annotated_image_rgb, caption='Ảnh kết quả phát hiện lỗi.', use_container_width=True)

                    # Hiển thị thông tin chi tiết (tùy chọn)
                    detected_objects = results[0].boxes.data.cpu().numpy() # Lấy thông tin boxes (xyxy, conf, cls)
                    if len(detected_objects) > 0:
                        st.subheader("Chi tiết các lỗi được phát hiện:")
                        detection_details = []
                        for i, det in enumerate(detected_objects):
                            x1, y1, x2, y2, conf, cls_id = det
                            class_name = model.names[int(cls_id)] # Lấy tên lớp từ model
                            detection_details.append(
                                f"- Lỗi {i+1}: **{class_name}** (Độ tin cậy: {conf:.2f}) tại vị trí [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
                            )
                        st.markdown("\n".join(detection_details))
                    else:
                        st.success("Không phát hiện thấy lỗi nào với ngưỡng tin cậy đã chọn.")

            else:
                st.error("Mô hình chưa được tải thành công, không thể phát hiện lỗi.")
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi xử lý ảnh: {e}")

else:
    st.info("Vui lòng tải lên một file ảnh để bắt đầu.")

# Chân trang (tùy chọn)
st.markdown("---")
st.caption("Phát triển bởi [Tên của bạn] sử dụng YOLOv5 và Streamlit.")