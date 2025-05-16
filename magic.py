from PIL import Image, ImageEnhance
import os

input_folder = "y"
output_folder = "z"
os.makedirs(output_folder, exist_ok=True)

def process_image_pillow(image_path, output_path):
    img = Image.open(image_path)
    enhancer = ImageEnhance.Brightness(img)
    dark_img = enhancer.enhance(0.2)  # 1.0 là gốc, <1.0 là tối
    dark_img.save(output_path)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        process_image_pillow(input_path, output_path)

print("Hoàn tất xử lý ảnh bằng Pillow.")