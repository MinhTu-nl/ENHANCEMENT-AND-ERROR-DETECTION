import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert binary masks to YOLO format')
    parser.add_argument('--mask_dir', type=str, default='screw/masks', help='Root directory containing mask folders')
    parser.add_argument('--output_dir', type=str, default='screw/labels',help='Output directory for YOLO format labels')
    parser.add_argument('--min_area', type=int, default=100,help='Minimum area for a defect to be considered (in pixels)')
    return parser.parse_args()

def get_class_id_from_path(mask_path):
    """Determine class ID based on mask path"""
    if 'manipulated_front' in mask_path:
        return 0
    elif 'scratch_head' in mask_path:
        return 1
    elif 'scratch_neck' in mask_path:
        return 2
    elif 'thread_side' in mask_path:
        return 3
    elif 'thread_top' in mask_path:
        return 4
    else:
        raise ValueError(f"Unknown defect type in path: {mask_path}")

def process_mask(mask_path, output_dir, min_area):
    """Process a single mask file and save YOLO format annotation"""
    try:
        # Read and threshold mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read mask {mask_path}")
            return
        
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        height, width = binary.shape
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Prepare output file
        base_name = os.path.basename(mask_path).replace('_mask.png', '.txt')
        txt_path = os.path.join(output_dir, base_name)
        
        class_id = get_class_id_from_path(mask_path)
        valid_defects = []
        
        # Process each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Convert to YOLO format (normalized center coordinates and dimensions)
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            norm_w = w / width
            norm_h = h / height
            
            # Ensure values are within [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            norm_w = max(0, min(1, norm_w))
            norm_h = max(0, min(1, norm_h))
            
            valid_defects.append((class_id, x_center, y_center, norm_w, norm_h))
        
        # Write to file if there are valid defects
        if valid_defects:
            with open(txt_path, 'w') as f:
                for defect in valid_defects:
                    f.write(f"{defect[0]} {defect[1]:.6f} {defect[2]:.6f} {defect[3]:.6f} {defect[4]:.6f}\n")
        else:
            # Create empty file if no valid defects found
            open(txt_path, 'a').close()
            
    except Exception as e:
        print(f"Error processing {mask_path}: {str(e)}")

def main():
    args = parse_arguments()
    
    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all mask files
    mask_paths = glob(os.path.join(args.mask_dir, '**/*_mask.png'), recursive=True)
    if not mask_paths:
        print(f"No mask files found in {args.mask_dir}")
        return
    
    print(f"Found {len(mask_paths)} mask files")
    print(f"Converting masks to YOLO format (min_area={args.min_area})...")
    
    # Process all masks with progress bar
    for mask_path in tqdm(mask_paths):
        process_mask(mask_path, args.output_dir, args.min_area)
    
    print(f"Conversion complete. YOLO format labels saved to {args.output_dir}")

if __name__ == '__main__':
    main()