import numpy as np
from .pixel_clustering import count_connected_pixels

def remove_noise_white(image, threshold):
    """Remove noise based on connected white pixel count."""
    h, w = image.shape
    result = image.copy()
    visited = np.zeros((h, w), dtype=bool)
    changes_made = False
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255 and not visited[y, x]:
                pixel_count, pixels = count_connected_pixels(image, y, x, visited, 255)
                print(f"White pixel cluster at ({y}, {x}): {pixel_count} pixels")
                if pixel_count < threshold:
                    for py, px in pixels:
                        result[py, px] = 0
                    changes_made = True
    return result, changes_made

def remove_noise_black(image, threshold):
    """Remove noise based on connected black pixel count."""
    h, w = image.shape
    result = image.copy()
    visited = np.zeros((h, w), dtype=bool)
    changes_made = False
    for y in range(h):
        for x in range(w):
            if image[y, x] == 0 and not visited[y, x]:
                pixel_count, pixels = count_connected_pixels(image, y, x, visited, 0)
                print(f"Black pixel cluster at ({y}, {x}): {pixel_count} pixels")
                if pixel_count < threshold:
                    for py, px in pixels:
                        result[py, px] = 255
                    changes_made = True
    return result, changes_made

def convert_black_to_white(image):
    """Convert black pixels to white at image boundaries."""
    h, w = image.shape
    result = image.copy()
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255:
                break
            if image[y, x] == 0:
                result[y, x] = 255
    for y in range(h):
        for x in range(w-1, -1, -1):
            if image[y, x] == 255:
                break
            if image[y, x] == 0:
                result[y, x] = 255
    for x in range(w):
        for y in range(h):
            if image[y, x] == 255:
                break
            if image[y, x] == 0:
                result[y, x] = 255
    for x in range(w):
        for y in range(h-1, -1, -1):
            if image[y, x] == 255:
                break
            if image[y, x] == 0:
                result[y, x] = 255
    return result