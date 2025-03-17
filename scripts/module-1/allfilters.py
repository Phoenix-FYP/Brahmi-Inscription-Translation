import cv2 as cv
import numpy as np  
import matplotlib
matplotlib.use('TkAgg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

# Load image and convert to grayscale
pic1 = cv.imread('./images/test.jpg')
img = cv.cvtColor(pic1, cv.COLOR_BGR2GRAY)

# Apply Gaussian Blur with different sigma values
gaussian_blur1 = cv.GaussianBlur(img, (5, 5), 1, cv.BORDER_DEFAULT)
gaussian_blur2 = cv.GaussianBlur(img, (5, 5), 7, cv.BORDER_DEFAULT)

# Apply Bilateral Filtering
bilateral_filtered = cv.bilateralFilter(img, 9, 75, 75)  # d=9, sigmaColor=75, sigmaSpace=75

# Otsu's Thresholding
ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
blur = cv.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Simple thresholding 
ret, thresh1 = cv.threshold(img, 135, 255, cv.THRESH_BINARY)

# Invert images so that text is black and background is white
img_inverted = cv.bitwise_not(img)
th2_inverted = cv.bitwise_not(th2)
th3_inverted = cv.bitwise_not(th3)
gaussian_blur1_inverted = cv.bitwise_not(gaussian_blur1)
gaussian_blur2_inverted = cv.bitwise_not(gaussian_blur2)
thresh1_inverted = cv.bitwise_not(thresh1)
bilateral_filtered_inverted = cv.bitwise_not(bilateral_filtered)  # Inverted bilateral filtering

# Display results
plt.figure(figsize=(18, 12))

plt.subplot(3, 3, 1)
plt.imshow(img_inverted, cmap='gray')
plt.title('Noisy Image (Inverted)')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(th2_inverted, cmap='gray')
plt.title("Otsu's Thresholding (Inverted)")
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(th3_inverted, cmap='gray')
plt.title("Otsu's Thresholding (Gaussian sigma=0, Inverted)")
plt.axis('off')

plt.subplot(3, 3, 4)
plt.imshow(gaussian_blur1_inverted, cmap='gray')
plt.title('Gaussian Blur (sigma=2, Inverted)')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(gaussian_blur2_inverted, cmap='gray')
plt.title('Gaussian Blur (sigma=7, Inverted)')
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(thresh1_inverted, cmap='gray')
plt.title('Simple Thresholding (130, 255, Inverted)')
plt.axis('off')

plt.subplot(3, 3, 7)
plt.imshow(bilateral_filtered, cmap='gray')
plt.title('Bilateral Filtering (Original)')
plt.axis('off')

plt.subplot(3, 3, 8)
plt.imshow(bilateral_filtered_inverted, cmap='gray')
plt.title('Bilateral Filtering (Inverted)')
plt.axis('off')

plt.tight_layout() 
plt.show()

# Save the Bilateral Filtered Image
cv.imwrite("bilateral_filtered.jpg", bilateral_filtered)
cv.imwrite("bilateral_filtered_inverted.jpg", bilateral_filtered_inverted)