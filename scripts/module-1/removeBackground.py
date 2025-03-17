
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image and convert to grayscale
image = cv.imread("./data/module-1/images/cleaned_image.png")
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 

# Define function to remove black background and make it white
def remove_black_background(img):
    """ Converts all black pixels outside the main content to white. """
    h, w = img.shape

    # Traverse pixels in four directions
    for i in range(h):
        # Left to Right
        for j in range(w):
            if img[i, j] == 0:
                img[i, j] = 255
            else:
                break  # Stop when hitting a white pixel
        
        # Right to Left
        for j in range(w-1, -1, -1):
            if img[i, j] == 0:
                img[i, j] = 255
            else:
                break  # Stop when hitting a white pixel

    for j in range(w):
        # Top to Bottom
        for i in range(h):
            if img[i, j] == 0:
                img[i, j] = 255
            else:
                break  # Stop when hitting a white pixel
        
        # Bottom to Top
        for i in range(h-1, -1, -1):
            if img[i, j] == 0:
                img[i, j] = 255
            else:
                break  # Stop when hitting a white pixel

    return img

# Apply background removal function on dilated image
cleaned_image = remove_black_background(gray_image)

# Save and Display the Result
cv.imwrite("./data/module-1/images/cleaned_image.png", cleaned_image)

plt.imshow(cleaned_image, cmap="gray")
plt.axis("off")
plt.title("Cleaned Image (Background Removed)")
plt.show()