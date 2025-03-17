import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image and convert to grayscale
image = cv.imread("./data/module-1/images/test0.jpg")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply Non-Local Means Denoising
denoised = cv.fastNlMeansDenoising(gray, None, 30, 7, 21)

# Apply Bilateral Filtering
bilateral_filtered = cv.bilateralFilter(denoised, 9, 75, 75)

# Apply Otsu's Thresholding
_, binary = cv.threshold(bilateral_filtered, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Invert the binary image
binary_inverted = cv.bitwise_not(binary)

# Apply **Dilation** to connect broken parts of letters
kernel = np.ones((3, 3), np.uint8)  # Structuring element (3x3 kernel)
dilated = cv.dilate(binary_inverted, kernel, iterations=2)  # Apply dilation

cv.imwrite("./data/module-1/images/cleaned_image.png", dilated)

# Display the final processed image
plt.figure(figsize=(10, 6))
plt.imshow(dilated, cmap="gray")
plt.title("Processed Image")
plt.axis("off")
plt.show()