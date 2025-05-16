import os
import shutil

# Đường dẫn đến folder chứa ảnh gốc
source_folder = "screw/images/scratch_head"
# Đường dẫn đến folder lưu ảnh sau khi copy
target_folder = "screw/images"

# Tạo thư mục đích nếu chưa có
os.makedirs(target_folder, exist_ok=True)

# Lấy danh sách ảnh và sắp xếp (để đúng thứ tự 000, 001,...)
image_files = sorted(f for f in os.listdir(source_folder) if f.lower().endswith('.png'))

count = 0

for image in image_files:
    image_path = os.path.join(source_folder, image)
    for i in range(5):  # Copy 5 lần
        new_name = f"scratch_head_{count:03d}.png"
        new_path = os.path.join(target_folder, new_name)
        shutil.copy(image_path, new_path)
        count += 1

print("Đã copy xong tất cả ảnh.")
