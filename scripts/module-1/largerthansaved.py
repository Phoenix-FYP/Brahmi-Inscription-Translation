import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import splprep, splev

# Load the image in grayscale 
image = cv.imread("./data/module-1/images/cleaned_image.png")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply Noise Reduction & Thresholding
denoised = cv.fastNlMeansDenoising(gray, None, 30, 7, 21)
bilateral_filtered = cv.bilateralFilter(denoised, 9, 75, 75)
_, binary = cv.threshold(bilateral_filtered, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Invert the binary image
binary_inverted = cv.bitwise_not(binary)

# Apply Dilation to connect broken parts of letters
kernel = np.ones((3, 3), np.uint8)
dilated = cv.dilate(binary_inverted, kernel, iterations=2)

# Find contours to detect the region of interest
contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Approximate the text baseline curve using the lowest contour points
contour_points = np.vstack([c[:, 0] for c in contours])  # Flatten contours
contour_points = contour_points[contour_points[:, 0].argsort()]  # Sort by X-coordinates

# Fit a B-spline curve to the detected text baseline
tck, u = splprep([contour_points[:, 0], contour_points[:, 1]], s=50)
new_x, new_y = splev(np.linspace(0, 1, 100), tck)  # Generate smooth curve

# Segment characters along the spline
char_bboxes = []  # Store character bounding boxes
for c in contours:
    x, y, w, h = cv.boundingRect(c)
    char_bboxes.append((x, y, w, h))

# Sort character boxes based on spline alignment
char_bboxes = sorted(char_bboxes, key=lambda b: b[0])  # Sort left to right

# Ensure output directory exists
output_dir = "cropped_characters"
os.makedirs(output_dir, exist_ok=True)

# Save only characters larger than 34x34
valid_char_bboxes = []
for idx, (x, y, w, h) in enumerate(char_bboxes):
    if w > 37 and h > 37:  # Check if character is larger than 34x34
        char_crop = binary[y:y+h, x:x+w]  # Crop the character region
        filename = os.path.join(output_dir, f"char_{idx+1}.png")
        cv.imwrite(filename, char_crop)
        valid_char_bboxes.append((x, y, w, h))  # Add valid bounding box

# Draw segmentation result
segmented_image = image.copy()
for (x, y, w, h) in valid_char_bboxes:
    cv.rectangle(segmented_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Draw the fitted spline curve
for i in range(len(new_x) - 1):
    cv.line(segmented_image, (int(new_x[i]), int(new_y[i])), (int(new_x[i+1]), int(new_y[i+1])), (255, 0, 0), 2)

# Save and Display Results
cv.imwrite("segmented_characters.png", segmented_image)

plt.figure(figsize=(10, 6))
plt.imshow(cv.cvtColor(segmented_image, cv.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Spline-Based Character Segmentation")
plt.show()

print(f"✅ {len(valid_char_bboxes)} characters were successfully saved in '{output_dir}'")