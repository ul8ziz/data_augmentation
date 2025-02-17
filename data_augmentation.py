import albumentations as A
import cv2
import os
import numpy as np
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def create_advanced_augmentation_pipeline():
    """
    Create a comprehensive and diverse augmentation pipeline
    with multiple transformation categories
    """
    transform = A.Compose([
        # Geometric Transformations (Advanced)
        A.OneOf([
            A.RandomRotate90(p=1),
            A.Rotate(limit=(-90, 90), p=1),
            A.Affine(
                scale=(0.8, 1.2),  # Scale between 80% and 120%
                translate_percent=(-0.1, 0.1),  # Translate up to 10%
                rotate=(-45, 45),  # Rotate between -45 and 45 degrees
                shear=(-15, 15),  # Shear between -15 and 15 degrees
                p=1
            )
        ], p=1),
        
        # Flipping and Mirroring
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1)
        ], p=0.7),
        
        # Perspective and Elastic Transformations
        A.OneOf([
            A.ElasticTransform(
                alpha=1, 
                sigma=50, 
                interpolation=cv2.INTER_LINEAR, 
                border_mode=cv2.BORDER_CONSTANT, 
                p=1
            ),
            A.Perspective(scale=(0.05, 0.1), p=1),
            A.OpticalDistortion(
                distort_limit=0.1, 
                interpolation=cv2.INTER_LINEAR, 
                p=1
            )
        ], p=0.5),
        
        # Color Space Transformations
        A.OneOf([
            A.ColorJitter(p=1),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1),
            A.HueSaturationValue(
                hue_shift_limit=20, 
                sat_shift_limit=30, 
                val_shift_limit=20, 
                p=1
            )
        ], p=0.8),
        
        # Brightness and Contrast
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=1
            ),
            A.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=1),
            A.Equalize(p=1)
        ], p=0.7),
        
        # Noise Augmentations
        A.OneOf([
            A.GaussNoise(
                mean=0, 
                std=25, 
                p=1
            ),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=1),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1)
        ], p=0.6),
        
        # Blur and Sharpen
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MedianBlur(blur_limit=7, p=1),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1)
        ], p=0.5),
        
        # Weather and Lighting Simulation
        A.OneOf([
            A.RandomFog(
                fog_coef=0.2,  # Single fog coefficient
                alpha_coef=0.1,  # Fog intensity
                p=1
            ),
            A.RandomRain(
                drop_length=10,  # Rain drop length
                drop_width=1,    # Rain drop width
                drop_color=(200, 200, 200),  # Rain drop color
                blur_value=1,    # Blur intensity
                p=1
            ),
            A.RandomSnow(
                snow_point=0.2,  # Snow density
                brightness_coeff=1.5,  # Snow brightness
                p=1
            )
        ], p=0.4)
    ], p=1)
    
    return transform

def augment_images(input_dir, output_dir, num_augmentations=50):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Track augmentation stats
    total_images_processed = 0
    total_augmented_images = 0
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(input_dir):
        # Create corresponding subdirectory in output folder
        relative_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        
        # Create output subdirectory if it doesn't exist
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        # Process each image in the current directory
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Read image
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path)
                
                if image is None:
                    logging.warning(f"Could not read image: {image_path}")
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                total_images_processed += 1
                
                # Generate augmented images
                for i in range(num_augmentations):
                    try:
                        # Get the augmentation pipeline
                        transform = create_advanced_augmentation_pipeline()
                        
                        # Apply transformations
                        augmented = transform(image=image)['image']
                        
                        # Save augmented image
                        output_filename = f"{os.path.splitext(filename)[0]}_aug_{i}{os.path.splitext(filename)[1]}"
                        output_path = os.path.join(output_subdir, output_filename)
                        cv2.imwrite(output_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
                        total_augmented_images += 1
                    except Exception as e:
                        logging.error(f"Error augmenting {image_path}: {e}")
    
    logging.info(f"Augmentation complete.")
    logging.info(f"Total images processed: {total_images_processed}")
    logging.info(f"Total augmented images generated: {total_augmented_images}")

if __name__ == "__main__":
    input_directory = "data"  # Your input data folder
    output_directory = "data_augmented"  # Where augmented images will be saved
    augment_images(input_directory, output_directory, num_augmentations=50)