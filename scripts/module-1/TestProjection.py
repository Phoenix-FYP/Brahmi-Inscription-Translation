import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the image in grayscale
image_path = "./data/module-1/images/cleaned_image.png"
gray = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

# Ensure output directory exists
output_dir = "cropped_characters"
os.makedirs(output_dir, exist_ok=True)

# Invert colors: Ensure text is black (0) and background is white (255)
_, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# Compute vertical projection
vertical_projection = np.sum(binary, axis=0)

# Plot the vertical projection histogram
plt.figure(figsize=(10, 4))
plt.plot(vertical_projection)
plt.title("Vertical Projection Profile")
plt.xlabel("X-axis (Columns)")
plt.ylabel("Sum of pixel intensities")
plt.show()

# Find character boundaries using projection profile
threshold = np.max(vertical_projection) * 0.1  # Adjust threshold as needed
split_positions = np.where(vertical_projection < threshold)[0]

# Extract individual character regions
char_images = []
prev = 0
for i in range(1, len(split_positions)):
    if split_positions[i] - split_positions[i-1] > 5:  # Avoid narrow gaps
        x_start, x_end = prev, split_positions[i]
        char_crop = binary[:, x_start:x_end]
        if char_crop.shape[1] > 10:  # Ignore small segments
            char_images.append(char_crop)
            filename = os.path.join(output_dir, f"char_{len(char_images)}.png")
            cv.imwrite(filename, char_crop)
        prev = x_end

# Display the segmented characters
fig, axes = plt.subplots(1, len(char_images), figsize=(15, 5))
for ax, char_img in zip(axes, char_images):
    ax.imshow(char_img, cmap='gray')
    ax.axis("off")
plt.show()