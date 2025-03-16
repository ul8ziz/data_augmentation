import os
import cv2
import numpy as np
import albumentations as A
from PIL import Image, ImageFont, ImageDraw, ImageOps
import random
import logging

# Configure logging
# تهيئة تسجيل
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

class CharacterAugmentation:
    """
    Initialize Character Augmentation Pipeline
    تهيئة خطوط تعزيز الحروف
    """
    def __init__(self, input_dir, output_dir, num_augmentations=1000):
        """
        Initialize Character Augmentation Pipeline
        تهيئة خطوط تعزيز الحروف
        
        :param input_dir: Directory containing input character images
        :param output_dir: Directory to save augmented images
        :param num_augmentations: Number of augmentations per image
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_augmentations = num_augmentations
        
        # Create output directory if it doesn't exist
        # إنشاء دليل الإخراج إذا لم يكن موجودًا
        os.makedirs(output_dir, exist_ok=True)
    
    def create_augmentation_pipeline(self):
        """
        Create a comprehensive augmentation pipeline for character recognition
        إنشاء خطوط تعزيز شاملة للتعرف على الحروف
        with subtle variations to maintain recognition quality
        مع اختلافات بسيطة للحفاظ على جودة التعرف
        
        :return: Albumentations composition of augmentations
        """
        return A.Compose([
            # Geometric Transformations (Subtle)
            # التحولات الهندسية (بشكل بسيط)
            A.OneOf([
                A.RandomRotate90(p=0.3),  # Reduced probability
                A.Rotate(limit=(-15, 15), border_mode=cv2.BORDER_REPLICATE, p=0.5),  # Reduced rotation range
                A.Affine(
                    scale=(0.9, 1.1),      # Reduced scale range
                    translate_percent=(-0.1, 0.1),  # Reduced translation
                    rotate=(-10, 10),      # Reduced rotation
                    shear=(-5, 5),         # Reduced shear
                    p=0.5
                )
            ], p=0.7),
            
            # Perspective and Elastic Transformations (Subtle)
            # تحولات المنظور والمرونة (بشكل بسيط)
            A.OneOf([
                A.ElasticTransform(
                    alpha=60,              # Reduced elasticity
                    sigma=60 * 0.05,
                    alpha_affine=60 * 0.03,
                    p=0.5
                ),
                A.Perspective(scale=(0.02, 0.08), keep_size=True, p=0.3),  # Reduced perspective
                A.OpticalDistortion(
                    distort_limit=0.1,     # Reduced distortion
                    shift_limit=0.1,
                    p=0.3
                )
            ], p=0.4),
            
            # Noise and Texture Augmentations (Subtle)
            # تعزيزات الضوضاء والملمس (بشكل بسيط)
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.4),  # Reduced noise
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=0.3),
                A.ImageCompression(quality_lower=90, quality_upper=100, p=0.3),  # Higher quality
                A.CoarseDropout(
                    max_holes=5,           # Reduced holes
                    max_height=2,
                    max_width=2,
                    min_holes=1,
                    min_height=1,
                    min_width=1,
                    p=0.3
                )
            ], p=0.4),
            
            # Blur and Sharpness (Subtle)
            # الضبابية والحدة (بشكل بسيط)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),  # Reduced blur
                A.MotionBlur(blur_limit=(3, 5), p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.0), p=0.4)
            ], p=0.4),
            
            # Contrast and Brightness (Subtle)
            # التباين والسطوع (بشكل بسيط)
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,   # Reduced brightness change
                    contrast_limit=0.1,     # Reduced contrast change
                    p=0.4
                ),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),  # Reduced CLAHE intensity
                A.RandomGamma(gamma_limit=(90, 110), p=0.3)  # Reduced gamma range
            ], p=0.4),
            
            # Border Augmentation (Subtle)
            # تعزيز الحدود (بشكل بسيط)
            A.OneOf([
                A.PadIfNeeded(
                    min_height=128,
                    min_width=128,
                    border_mode=cv2.BORDER_REPLICATE,
                    p=0.3
                ),
                A.PadIfNeeded(
                    min_height=128,
                    min_width=128,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=255,  # White background for better visibility
                    p=0.3
                )
            ], p=0.2)
        ], p=1)
    
    def generate_synthetic_character(self, char, font_path=None, font_size=100, image_size=(128, 128)):
        """
        Generate a synthetic character image with improved positioning
        إنشاء صورة حرف اصطناعية مع تحسين الموضع
        
        :param char: Character to generate
        :param font_path: Path to font file (optional)
        :param font_size: Size of the font
        :param image_size: Size of the output image
        :return: Numpy array of the character image
        """
        # Create a white background image
        # إنشاء صورة بخلفية بيضاء
        image = Image.new('L', image_size, color=255)
        draw = ImageDraw.Draw(image)
        
        # Use a default font if no font path is provided
        # استخدام خط افتراضي إذا لم يتم توفير مسار الخط
        if font_path is None:
            try:
                # Try to use a system font
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                # Fallback to default font
                font = ImageFont.load_default()
        else:
            font = ImageFont.truetype(font_path, font_size)
        
        # Get font metrics
        # الحصول على مقاييس الخط
        left, top, right, bottom = font.getbbox(char)
        text_width = right - left
        text_height = bottom - top
        
        # Calculate text position to center it
        # حساب موضع النص لتوسيطه
        position = (
            (image_size[0] - text_width) // 2, 
            (image_size[1] - text_height) // 2
        )
        
        # Draw the character
        # رسم الحرف
        draw.text(position, char, font=font, fill=0)
        
        # Optional: Add slight random rotation or noise
        # اختياري: إضافة دوران عشوائي بسيط أو ضوضاء
        if random.random() < 0.3:
            image = image.rotate(random.uniform(-5, 5), fillcolor=255)
        
        return np.array(image)
    
    def augment_images(self):
        """
        Perform augmentation on all images in the input directory
        تنفيذ تعزيز على جميع الصور في الدليل المدخل
        
        :return: None
        """
        # Track augmentation stats
        # تتبع إحصائيات التعزيز
        total_images_processed = 0
        total_augmented_images = 0
        
        # Walk through all subdirectories
        # التنقل عبر جميع الأدلة الفرعية
        for root, dirs, files in os.walk(self.input_dir):
            # Create corresponding subdirectory in output folder
            # إنشاء الدليل الفرعي المقابل في مجلد الإخراج
            relative_path = os.path.relpath(root, self.input_dir)
            output_subdir = os.path.join(self.output_dir, relative_path)
            
            # Create output subdirectory if it doesn't exist
            # إنشاء الدليل الفرعي للإخراج إذا لم يكن موجودًا
            os.makedirs(output_subdir, exist_ok=True)
            
            # Process each image in the current directory
            # معالجة كل صورة في الدليل الحالي
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                    # Read image
                    # قراءة الصورة
                    image_path = os.path.join(root, filename)
                    
                    try:
                        # Read image in grayscale
                        # قراءة الصورة بتدرج رمادي
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        
                        if image is None:
                            logging.warning(f"Could not read image: {image_path}")
                            continue
                        
                        # Resize image to a standard size if needed
                        # تغيير حجم الصورة إلى حجم قياسي إذا لزم الأمر
                        image = cv2.resize(image, (128, 128))
                        
                        total_images_processed += 1
                        
                        # Get the augmentation pipeline
                        # الحصول على خطوط التعزيز
                        transform = self.create_augmentation_pipeline()
                        
                        # Generate augmented images
                        # إنشاء الصور المعززة
                        for i in range(self.num_augmentations):
                            try:
                                # Apply transformations
                                # تطبيق التحولات
                                augmented = transform(image=image)['image']
                                
                                # Save augmented image
                                # حفظ الصورة المعززة
                                output_filename = f"{os.path.splitext(filename)[0]}_aug_{i}{os.path.splitext(filename)[1]}"
                                output_path = os.path.join(output_subdir, output_filename)
                                cv2.imwrite(output_path, augmented)
                                total_augmented_images += 1
                            
                            except Exception as e:
                                logging.error(f"Error augmenting {filename}: {e}")
                    
                    except Exception as e:
                        logging.error(f"Error processing {filename}: {e}")
        
        logging.info(f"Augmentation complete.")
        logging.info(f"Total images processed: {total_images_processed}")
        logging.info(f"Total augmented images generated: {total_augmented_images}")
    
    def generate_synthetic_dataset(self, characters='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', num_samples=1000):
        """
        Generate a synthetic dataset of characters
        إنشاء مجموعة بيانات اصطناعية من الحروف
        
        :param characters: String of characters to generate
        :param num_samples: Number of synthetic images to generate
        """
        # Create output directory for synthetic data
        # إنشاء دليل الإخراج للبيانات الاصطناعية
        synthetic_dir = os.path.join(self.output_dir, 'synthetic')
        os.makedirs(synthetic_dir, exist_ok=True)
        
        # Generate synthetic images
        # إنشاء الصور الاصطناعية
        for _ in range(num_samples):
            # Randomly select a character
            # اختيار حرف بشكل عشوائي
            char = random.choice(characters)
            
            # Generate synthetic character
            # إنشاء حرف اصطناعي
            synthetic_image = self.generate_synthetic_character(char)
            
            # Save the synthetic image
            # حفظ الصورة الاصطناعية
            filename = f"synthetic_{char}_{_}.png"
            cv2.imwrite(os.path.join(synthetic_dir, filename), synthetic_image)
        
        logging.info(f"Generated {num_samples} synthetic character images")

def main():
    # Full path to the data directory
    # المسار الكامل لدليل البيانات
    input_directory = r"D:\ul8ziz\GitHub\data_Augmentation\data"  # Directory with input character images
    output_directory = r"D:\ul8ziz\GitHub\data_Augmentation\data_augmented0"  # Output directory for augmented images
    
    # Initialize augmentation
    # تهيئة التعزيز
    augmenter = CharacterAugmentation(input_directory, output_directory, num_augmentations=1000)
    
    # Perform image augmentation
    # تنفيذ تعزيز الصور
    augmenter.augment_images()
    
    # Generate synthetic characters (optional)
    # إنشاء أحرف اصطناعية (اختياري)
    augmenter.generate_synthetic_dataset()

if __name__ == "__main__":
    main()
