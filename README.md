# Data Augmentation Project

## Overview
This project provides a robust data augmentation pipeline for image processing, utilizing the powerful `albumentations` library. It helps increase the diversity and size of image datasets by applying various transformations, which can improve machine learning model performance and generalization.

## Features
- Advanced geometric transformations
- Random rotations
- Scaling and translation
- Shearing
- Flipping and mirroring
- Configurable augmentation pipeline

## Requirements
- Python 3.7+
- OpenCV (`cv2`)
- Albumentations
- NumPy

## Installation
```bash
pip install albumentations opencv-python numpy
```

## Usage
1. Place your input images in the `data/` directory
2. Run the augmentation script:
```bash
python data_augmentation.py
```

### Customization
- Modify `num_augmentations` to control the number of augmented images
- Adjust the augmentation pipeline in `create_advanced_augmentation_pipeline()` to suit your needs

## Augmentation Techniques
- Rotation (0-90 degrees)
- Scaling (80-120%)
- Translation (±10%)
- Shearing (±15 degrees)
- Horizontal/Vertical flipping

## Output
Augmented images are saved in the `data_augmented/` directory

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
