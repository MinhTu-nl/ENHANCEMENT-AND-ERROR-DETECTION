AI-Powered Defect Detection in Low-Light Environments
Introduction
This project develops an AI-based application for detecting defects in various products (e.g., cables, bottles, screws, transistors, zippers) under low-light conditions. By combining a Generative Adversarial Network (GAN) for image enhancement and YOLOv11n for object detection, the system improves the visibility of low-light images and accurately identifies defects. The application is built with Streamlit for an interactive user interface, enabling users to upload images, process them, and view defect detection results.
The project addresses the challenge of inspecting products in environments with poor lighting, which is common in industrial settings. It provides a robust solution for quality control by enhancing image quality and detecting defects with high precision.
Processing Workflow
The project follows a structured pipeline to process images and detect defects:

Image Upload: Users upload an image of a product (JPG, PNG, or JPEG) through the Streamlit interface.
Image Enhancement:
If the "Enhance Lighting" option is enabled, the GAN-based Generator model processes the low-light image to improve brightness and contrast.
The enhanced image is resized to 256x256 pixels for compatibility with the GAN model.


Defect Detection:
The enhanced (or original, if enhancement is disabled) image is fed into a product-specific YOLOv11n model.
The YOLO model detects defects and annotates them with bounding boxes and class labels, based on a user-defined confidence threshold.


Result Display:
The application displays the original image, enhanced image (if applicable), and the annotated image with detected defects.
Detailed defect information (class, confidence, and bounding box coordinates) is provided.


History and Downloads:
Results are stored in a session history for review.
Users can download the enhanced and annotated images.


Dataset Preparation and Model Training (Backend):
Scripts are provided to preprocess datasets, convert annotations to YOLO format, and train both GAN and YOLO models.



Purpose and Usage of Files
Below is a description of each file in the project, outlining its purpose and how it is used:

1_convert_masks_to_yolo.py:

Purpose: Converts binary mask images into YOLO-compatible annotation files for defect detection.
Usage: Processes mask files in the specified directory (screw/masks by default), extracts contours, and generates normalized bounding box annotations in YOLO format. Outputs .txt files to the specified output directory (screw/labels).
Key Features:
Filters defects based on a minimum area threshold.
Assigns class IDs based on mask file paths (e.g., manipulated_front → class 0).
Handles errors gracefully and creates empty annotation files for masks with no valid defects.


Command Example: python 1_convert_masks_to_yolo.py --mask_dir screw/masks --output_dir screw/labels --min_area 100


2_prepare_dataset.py:

Purpose: Organizes images and YOLO annotations into a structured dataset for training, with train/validation splits.
Usage: Takes image and label directories as input, validates their consistency, and creates a YOLO dataset structure (screw/defects by default) with images and labels subdirectories for train (and optionally validation) sets. Generates a dataset.yaml configuration file.
Key Features:
Supports symbolic links or file copying to save disk space.
Splits dataset based on a validation ratio (default 20%).
Validates that each label has a corresponding image (except for "good" products).


Command Example: python 2_prepare_dataset.py --images_dir screw/images --labels_dir screw/labels --output_dir screw/defects --val_ratio 0.2


3_train_model.py:

Purpose: Trains a YOLOv11n model for defect detection using the Ultralytics YOLO library.
Usage: Loads a dataset configuration (dataset.yaml), initializes a YOLO model with specified weights (default yolo11n.pt), and trains it with customizable parameters (e.g., image size, epochs, batch size).
Key Features:
Updates relative paths in dataset.yaml to absolute paths for compatibility.
Supports training on GPU or CPU.
Saves training results to a specified project directory (weights/Yolo11n by default).


Command Example: python 3_train_model.py --data screw/defects/dataset.yaml --weights yolo11n.pt --imgsz 640 --epochs 200


model.py:

Purpose: Defines the GAN architecture (Generator, Discriminator) and a PerceptualLoss module for low-light image enhancement.
Usage: Used by train.py and app.py to enhance low-light images. The Generator uses a U-Net-like architecture with skip connections, while the Discriminator evaluates image realism. PerceptualLoss leverages a pretrained VGG16 model for feature-based loss.
Key Features:
Initializes weights with normal distribution for stability.
Generator enhances images from 3-channel input to 3-channel output.
PerceptualLoss improves enhancement quality by comparing VGG16 features.




utils.py:

Purpose: Provides utility functions and a dataset class for loading and processing low-light and high-light image pairs.
Usage: Used by train.py to create data loaders for GAN training and by app.py for image saving. The LoLDataset class loads paired images, applies transformations, and resizes them to a target size (default 256x256).
Key Features:
Converts images to PyTorch tensors with normalization.
Saves enhanced images to disk in a compatible format.
Ensures robust image loading and color space conversion.




train.py:

Purpose: Trains the GAN model for low-light image enhancement.
Usage: Loads training and validation datasets, trains the Generator and Discriminator using adversarial, pixel, and perceptual losses, and saves checkpoints and evaluation images periodically.
Key Features:
Supports resuming training from a checkpoint.
Uses Adam optimizer with customizable learning rate and betas.
Saves enhanced images for visual evaluation every 10 epochs.


Command Example: python train.py (uses default parameters or specified checkpoint)


app.py:

Purpose: Implements the Streamlit web application for interactive defect detection.
Usage: Runs a user interface where users select a product, upload an image, adjust settings (e.g., confidence threshold, lighting enhancement), and view results. Integrates the GAN model for enhancement and YOLO models for defect detection.
Key Features:
Supports multiple product types with pretrained YOLO models.
Displays original, enhanced, and annotated images side-by-side.
Maintains a processing history and allows downloading results.
Provides a user-friendly sidebar for configuration.


Command Example: streamlit run app.py



Conclusion
This project delivers a powerful solution for defect detection in low-light environments, combining state-of-the-art deep learning models (GAN and YOLOv11n) with a user-friendly interface. The modular codebase supports dataset preparation, model training, and real-time inference, making it adaptable for various industrial applications. By addressing the challenges of low-light imaging and precise defect detection, the system enhances quality control processes and demonstrates the potential of AI in manufacturing.
Future improvements could include:

Optimizing GAN training for faster convergence.
Expanding YOLO model support for additional product types.
Adding real-time video processing capabilities.
Deploying the application on a cloud platform for scalability.

For questions or contributions, please contact Nguyen Le Minh Tu (@2025 Copyright).
