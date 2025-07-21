import numpy as np
import cv2 as cv

def create_irregular_crop(image, black_cluster_pixels):
    """Create an irregularly cropped image based on black cluster pixels."""
    h, w = image.shape
    cropped_image = np.ones((h, w), dtype=np.uint8) * 255
    if black_cluster_pixels:
        y_coords, x_coords = zip(*black_cluster_pixels)
        max_y = max(y_coords)
        min_x, max_x = 0, w - 1
        cropped_image[0:max_y+1, min_x:max_x+1] = image[0:max_y+1, min_x:max_x+1]
    return cropped_image