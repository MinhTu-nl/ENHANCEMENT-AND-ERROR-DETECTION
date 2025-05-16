import os
import yaml
import argparse
import random
from glob import glob
from tqdm import tqdm
import shutil

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prepare YOLOv5 dataset')
    parser.add_argument('--images_dir', type=str, default='screw/images', help='Directory containing original images')
    parser.add_argument('--labels_dir', type=str, default='screw/labels', help='Directory containing YOLO format labels')
    parser.add_argument('--output_dir', type=str, default='screw/defects', help='Output directory for YOLO dataset')
    parser.add_argument('--symlink', action='store_true',help='Use symbolic links instead of copying files')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio (0.0 means use same as train)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting dataset')
    return parser.parse_args()

def validate_dataset(labels_dir, images_dir):
    """Validate that each label file has corresponding image"""
    print("Validating dataset...")
    missing_images = 0
    missing_labels = 0
    
    # Check all label files have corresponding images
    label_files = glob(os.path.join(labels_dir, '*.txt'))
    for label_file in tqdm(label_files):
        base_name = os.path.basename(label_file).replace('.txt', '.png')
        image_path = os.path.join(images_dir, base_name)
        if not os.path.exists(image_path):
            missing_images += 1
    
    # Check all image files have corresponding labels (except good products)
    image_files = glob(os.path.join(images_dir, '*.png'))
    for image_file in tqdm(image_files):
        if 'good' in image_file.lower():
            continue  # Skip good products as they don't need labels
        base_name = os.path.basename(image_file).replace('.png', '.txt')
        label_path = os.path.join(labels_dir, base_name)
        if not os.path.exists(label_path):
            missing_labels += 1
    
    if missing_images > 0 or missing_labels > 0:
        print(f"Warning: Found {missing_images} missing images and {missing_labels} missing labels")
    else:
        print("Dataset validation passed - all files are matched")

def create_symlink_or_copy(src, dst, use_symlink=True):
    """Create symbolic link or copy file based on OS support"""
    if os.path.exists(dst):
        return
    
    src = os.path.abspath(src)
    if use_symlink:
        try:
            os.symlink(src, dst)
        except OSError:
            # Fallback to copy if symlink fails (e.g., on Windows without permission)
            shutil.copyfile(src, dst)
    else:
        shutil.copyfile(src, dst)

def split_dataset(image_files, val_ratio, seed=42):
    """Split dataset into train and validation sets"""
    if val_ratio <= 0:
        return image_files, image_files  # Use all for both
    
    random.seed(seed)
    random.shuffle(image_files)
    split_idx = int(len(image_files) * (1 - val_ratio))
    return image_files[:split_idx], image_files[split_idx:]

def prepare_dataset_structure(args):
    """Main function to prepare YOLOv5 dataset structure"""
    # Create directory structure
    os.makedirs(os.path.join(args.output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'labels', 'train'), exist_ok=True)
    
    if args.val_ratio > 0:
        os.makedirs(os.path.join(args.output_dir, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'labels', 'val'), exist_ok=True)
    
    # Get all image files (exclude good products if needed)
    image_files = glob(os.path.join(args.images_dir, '*.png'))
    
    # Split dataset if needed
    train_files, val_files = split_dataset(image_files, args.val_ratio, args.seed)
    
    print(f"Preparing dataset with {len(train_files)} training samples")
    if args.val_ratio > 0:
        print(f"and {len(val_files)} validation samples")
    
    # Process training set
    print("Processing training set...")
    for img_path in tqdm(train_files):
        base_name = os.path.basename(img_path)
        
        # Handle image
        dst_img = os.path.join(args.output_dir, 'images', 'train', base_name)
        create_symlink_or_copy(img_path, dst_img, args.symlink)
        
        # Handle label
        label_name = base_name.replace('.png', '.txt')
        src_label = os.path.join(args.labels_dir, label_name)
        if os.path.exists(src_label):
            dst_label = os.path.join(args.output_dir, 'labels', 'train', label_name)
            create_symlink_or_copy(src_label, dst_label, args.symlink)
    
    # Process validation set if exists
    if args.val_ratio > 0:
        print("Processing validation set...")
        for img_path in tqdm(val_files):
            base_name = os.path.basename(img_path)
            
            # Handle image
            dst_img = os.path.join(args.output_dir, 'images', 'val', base_name)
            create_symlink_or_copy(img_path, dst_img, args.symlink)
            
            # Handle label
            label_name = base_name.replace('.png', '.txt')
            src_label = os.path.join(args.labels_dir, label_name)
            if os.path.exists(src_label):
                dst_label = os.path.join(args.output_dir, 'labels', 'val', label_name)
                create_symlink_or_copy(src_label, dst_label, args.symlink)
    
    # Create dataset.yaml file
    create_dataset_yaml(args)

def create_dataset_yaml(args):
    """Create YOLOv5 dataset.yaml configuration file"""
    yaml_content = {
        'path': os.path.abspath(args.output_dir),
        'train': 'images/train',
        'val': 'images/val' if args.val_ratio > 0 else 'images/train',
        'names': {
            0: 'manipulated_front',
            1: 'scratch_head',
            2: 'scratch_neck',
            3: 'thread_side',
            4: 'thread_top'
        },
        'nc': 4,  # Số lượng classes
        'download': None
    }
    
    yaml_path = os.path.join(args.output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    
    print(f"Dataset configuration saved to {yaml_path}")

def main():
    args = parse_arguments()
    
    # Validate dataset first
    validate_dataset(args.labels_dir, args.images_dir)
    
    # Prepare dataset structure
    prepare_dataset_structure(args)
    
    print("Dataset preparation completed successfully")

if __name__ == '__main__':
    main()