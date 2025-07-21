import cv2 as cv
import numpy as np
import os

def load_and_preprocess_image(image_path):
    """Load an image, convert to grayscale, and invert it."""
    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image '{image_path}' not found")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    binary_inverted = cv.bitwise_not(gray)
    return image, gray, binary_inverted

def save_image(image, output_path):
    """Save an image to the specified path."""
    cv.imwrite(output_path, image)

def create_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)