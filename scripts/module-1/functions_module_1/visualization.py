import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.interpolate import splprep, splev

def plot_image_processing_results(binary_inverted, cleaned_image, cropped_image, denoised_image, 
                                 binary_inverted_denoised, final_image, final_image_second_pass, 
                                 black_clusters, largest_black_cluster, clusters, final_clusters, width):
    """Plot all image processing stages with bounding boxes."""
    plt.figure(figsize=(20, 14))
    
    plt.subplot(2, 4, 1)
    plt.imshow(binary_inverted, cmap="gray")
    plt.title("Binary Inverted Image")
    plt.axis("off")
    
    plt.subplot(2, 4, 2)
    plt.imshow(cleaned_image, cmap="gray")
    plt.title("Cleaned Image")
    if black_clusters and largest_black_cluster:
        min_y, max_y = largest_black_cluster['bounding_box'][1], largest_black_cluster['bounding_box'][3]
        min_x, max_x = 0, width - 1
        width_rect = max_x - min_x
        height = max_y - min_y
        rect_largest = patches.Rectangle(
            (min_x, min_y), width_rect, height,
            linewidth=2, edgecolor='green', facecolor='none', label='Region')
        plt.gca().add_patch(rect_largest)
        plt.text(min_x, min_y - 5, 'Region', color='green', fontsize=8)
    plt.axis("off")
    
    plt.subplot(2, 4, 3)
    plt.imshow(cropped_image, cmap="gray")
    plt.title("Irregularly Cropped Largest Cluster")
    if clusters and largest_black_cluster:
        max_y = largest_black_cluster['bounding_box'][3]
        min_x, max_x = 0, width - 1
        min_y = 0
        width_rect = max_x - min_x
        height = max_y - min_y
        rect_largest = patches.Rectangle(
            (min_x, min_y), width_rect, height,
            linewidth=2, edgecolor='blue', facecolor='none', label='Region'
        )
        plt.gca().add_patch(rect_largest)
        plt.text(min_x, min_y - 5, 'Region', color='blue', fontsize=8)
        plt.legend()
    plt.axis("off")
    
    plt.subplot(2, 4, 4)
    plt.imshow(denoised_image, cmap="gray")
    plt.title("Denoised Image (First Pass)")
    plt.axis("off")
    
    plt.subplot(2, 4, 5)
    plt.imshow(binary_inverted_denoised, cmap="gray")
    plt.title("Inverted Denoised Image")
    plt.axis("off")
    
    plt.subplot(2, 4, 6)
    plt.imshow(final_image, cmap="gray")
    plt.title("Final Image (After Black-to-White Conversion)")
    plt.axis("off")
    
    plt.subplot(2, 4, 7)
    plt.imshow(final_image_second_pass, cmap='gray')
    plt.title("Final Image (Second Pass - Black Noise Removal)")
    for cluster in final_clusters:
        min_x, min_y, max_x, max_y = cluster['bounding_box']
        char_index = cluster['char_index']
        is_sub_character = cluster['is_sub_character']
        width_rect = max_x - min_x
        height = max_y - min_y
        edgecolor = 'red' if is_sub_character else 'green'
        rect = patches.Rectangle(
            (min_x, min_y), width_rect, height,
            linewidth=1, edgecolor=edgecolor, facecolor='none'
        )
        plt.gca().add_patch(rect)
        plt.text(min_x, min_y - 5, f'Char {char_index}', color=edgecolor, fontsize=8)
    plt.axis("off")
    
    plt.savefig('image_processing_results.png')
    plt.tight_layout()
    plt.show()

def plot_labeled_image(final_image_second_pass, final_clusters):
    """Plot the final image with labeled bounding boxes and spline."""
    plt.figure(figsize=(10, 7))
    plt.imshow(final_image_second_pass, cmap='gray')
    plt.title("Labeled Image with Bounding Boxes")
    for cluster in final_clusters:
        min_x, min_y, max_x, max_y = cluster['bounding_box']
        char_index = cluster['char_index']
        is_sub_character = cluster['is_sub_character']
        width = max_x - min_x
        height = max_y - min_y
        edgecolor = 'red' if is_sub_character else 'green'
        rect = patches.Rectangle(
            (min_x, min_y), width, height,
            linewidth=1, edgecolor=edgecolor, facecolor='none'
        )
        plt.gca().add_patch(rect)
        plt.text(min_x, min_y - 5, f'Char {char_index}', color=edgecolor, fontsize=8)
    if len(final_clusters) >= 4:
        centers = [cluster['center'] for cluster in final_clusters]
        x_centers, y_centers = zip(*centers)
        tck, u = splprep([x_centers, y_centers], s=10)
        u_fine = np.linspace(0, 1, 50)
        x_spline, y_spline = splev(u_fine, tck)
        plt.plot(x_spline, y_spline, 'b-', linewidth=1, label='spline')
        plt.legend()
    plt.axis("off")
    plt.savefig('labeled_bounding_boxes_standalone.png')
    plt.close()