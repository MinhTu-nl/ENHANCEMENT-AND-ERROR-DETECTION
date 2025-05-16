import os
import argparse
import yaml
from ultralytics import YOLO # Import thư viện YOLO từ Ultralytics

def parse_arguments():
    """Phân tích các đối số dòng lệnh."""
    parser = argparse.ArgumentParser(description='Start YOLOv5 training session using ultralytics library.')
    parser.add_argument('--data', type=str, default='cable/defects/dataset.yaml', help='Path to the dataset.yaml file.')
    parser.add_argument('--weights', type=str, default='yolo11n.pt', help='Initial weights path (e.g., yolov5s.pt, yolov5m.pt, or path to custom weights).')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='Train, val image size (pixels).')
    parser.add_argument('--batch-size', type=int, default=16, help='Total batch size for all GPUs.')
    parser.add_argument('--epochs', type=int, default=200, help='Total training epochs.')
    parser.add_argument('--project', default='weights/Yolo11n', help='Save results to project/name.')
    parser.add_argument('--name', default='defect_detection_exp', help='Save results to project/name.')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu.')
    args = parser.parse_args()
    return args

def check_data_path(data_yaml_path):
    """Kiểm tra và cập nhật đường dẫn tuyệt đối trong file data.yaml nếu cần."""
    try:
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Kiểm tra xem 'path' có phải là đường dẫn tương đối không
        if 'path' in data and not os.path.isabs(data['path']):
            # Chuyển đổi thành đường dẫn tuyệt đối dựa trên vị trí file yaml
            abs_path = os.path.abspath(os.path.join(os.path.dirname(data_yaml_path), data['path']))
            print(f"Updating relative path in {data_yaml_path} to absolute path: {abs_path}")
            data['path'] = abs_path
            # Ghi lại file yaml với đường dẫn tuyệt đối
            with open(data_yaml_path, 'w') as f:
                yaml.dump(data, f, sort_keys=False)
        elif 'path' not in data:
            print(f"Warning: 'path' key not found in {data_yaml_path}. Assuming paths inside are relative to CWD or absolute.")

    except Exception as e:
        print(f"Error processing {data_yaml_path}: {e}")
        print("Please ensure the YAML file is valid and the paths are correct.")
        # Có thể quyết định dừng script ở đây nếu file data quan trọng
        # exit(1)


if __name__ == '__main__':
    args = parse_arguments()
    check_data_path(args.data)
    model = YOLO(args.weights)

    print(f"Starting training with the following parameters:")
    print(f" > Data YAML: {args.data}")
    print(f" > Model/Weights: {args.weights}")
    print(f" > Image Size: {args.imgsz}")
    print(f" > Epochs: {args.epochs}")
    print(f" > Batch Size: {args.batch_size}")
    print(f" > Project: {args.project}")
    print(f" > Name: {args.name}")
    print(f" > Device: {'cuda' if args.device == '' or 'cuda' in args.device else 'cpu'}") # Ước tính thiết bị

    # 2. Thực hiện huấn luyện
    try:
        results = model.train(
            data=args.data,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch_size,
            project=args.project,
            name=args.name,
            device=args.device if args.device else None, # Truyền device nếu được chỉ định
            exist_ok=False # Đặt là True nếu muốn ghi đè lên lần chạy trước có cùng tên
        )
        print("Training completed successfully.")
        print(f"Results saved to: {results.save_dir}") # In đường dẫn thư mục kết quả

    except Exception as e:
        print(f"An error occurred during training: {e}")
    