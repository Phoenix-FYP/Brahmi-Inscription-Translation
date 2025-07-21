import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path, output_dir="./images", bias=20, kernel_size=(1, 1), dilation_iterations=1):
    """
    Preprocess an image with denoising, bilateral filtering, Otsu's thresholding, and dilation.
    
    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save intermediate images.
        bias (int): Value to subtract from Otsu's threshold for adjustment.
        kernel_size (tuple): Size of the dilation kernel.
        dilation_iterations (int): Number of dilation iterations.
    
    Returns:
        tuple: Original image, preprocessed dilated image, and intermediate images for visualization.
    """
    # Load image and convert to grayscale
    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image '{image_path}' not found")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply Non-Local Means Denoising
    denoised = cv.fastNlMeansDenoising(gray, None, 30, 7, 21)

    # Apply Bilateral Filtering
    bilateral_filtered = cv.bilateralFilter(denoised, 9, 75, 75)

    # Apply Otsu's Thresholding
    otsu_threshold, binary_otsu = cv.threshold(bilateral_filtered, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # printshe = cv.threshold(denoised, max(0, otsu_threshold - bias), 255, cv.THRESH_BINARY)
    # binary_otsu_adjusted = binary_otsu  # Since we apply bias directly, binary_otsu is the adjusted binary image
    adjusted_threshold = max(0, otsu_threshold - bias)  # Ensure threshold doesn't go below 0
    _, binary_otsu_adjusted = cv.threshold(denoised, adjusted_threshold, 255, cv.THRESH_BINARY)
    # Invert the binary image
    binary_otsu_adjusted_inverted = cv.bitwise_not(binary_otsu_adjusted)

    # Apply Dilation
    kernel = np.ones(kernel_size, np.uint8)
    dilated_otsu_adjusted = cv.dilate(binary_otsu_adjusted_inverted, kernel, iterations=dilation_iterations)

    # Save the final processed image
    output_path = f"{output_dir}/cleaned_image_otsu_adjusted.png"
    cv.imwrite(output_path, dilated_otsu_adjusted)

    # Return images for processing and visualization
    return image, dilated_otsu_adjusted, {
        'gray': gray,
        'denoised': denoised,
        'binary_otsu_adjusted': binary_otsu_adjusted,
        'binary_otsu_adjusted_inverted': binary_otsu_adjusted_inverted,
        'dilated_otsu_adjusted': dilated_otsu_adjusted
    }

def visualize_preprocessing_steps(image, intermediate_images, output_file='preprocessing_steps_adjusted_otsu.png'):
    """Visualize the preprocessing steps."""
    plt.figure(figsize=(15, 6))

    plt.subplot(2, 4, 1)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.imshow(intermediate_images['gray'], cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.imshow(intermediate_images['denoised'], cmap="gray")
    plt.title("Denoised Image")
    plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.imshow(intermediate_images['binary_otsu_adjusted'], cmap="gray")
    plt.title("Adjusted Otsu")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.imshow(intermediate_images['binary_otsu_adjusted_inverted'], cmap="gray")
    plt.title("Inverted (Adjusted Otsu)")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.imshow(intermediate_images['dilated_otsu_adjusted'], cmap="gray")
    plt.title("Dilated (Adjusted Otsu)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()