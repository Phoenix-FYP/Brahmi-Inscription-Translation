import numpy as np
import cv2 as cv
from .preprocessing import preprocess_image, visualize_preprocessing_steps
from .image_io import save_image, create_output_directory
from .pixel_clustering import analyze_black_pixel_clusters, merge_vertical_clusters, merge_contained_clusters, merge_close_horizontal_clusters
from .noise_removal import remove_noise_white, remove_noise_black, convert_black_to_white
from .image_cropping import create_irregular_crop
from .clutser_analysis import analyze_cluster_statistics, calculate_character_widths_and_z_scores, process_final_clusters
from .visualization import plot_image_processing_results, plot_labeled_image

def process_image(image_path: str, output_dir: str = "./cropped_final_characters", image_no: int = 1, threshold: int = 2000, output_base2:str = "../results/module-1/") -> dict:
    THRESHOLD = threshold
    max_iterations = 10

    create_output_directory(output_dir)

    # Step 1: Preprocess
    original_image, binary_inverted, intermediate_images = preprocess_image(
        image_path,
        output_dir=output_dir,
        bias=20,
        kernel_size=(1, 1),
        dilation_iterations=1
    )
    # visualize_preprocessing_steps(original_image, intermediate_images)

    image = cv.imread(f"{output_dir}/cleaned_image_otsu_adjusted.png")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    binary_inverted = cv.bitwise_not(gray)
    save_image(binary_inverted, f"{output_dir}/dilated.png")
    save_image(binary_inverted, f"{output_base2}/dilated.png")

    height, width = binary_inverted.shape
    black_clusters, average_size = analyze_black_pixel_clusters(binary_inverted)

    cleaned_image = binary_inverted
    save_image(cleaned_image, f"{output_dir}/cleaned_image.png")
    save_image(cleaned_image, f"{output_base2}/cleaned_image.png")

    if black_clusters:
        largest_black_cluster = max(black_clusters, key=lambda cluster: cluster['size'])
        cropped_image = create_irregular_crop(cleaned_image, largest_black_cluster['pixels'])
        save_image(cropped_image, f"{output_dir}/cropped_image.png")
        save_image(cropped_image, f"{output_base2}/cropped_image.png")
    else:
        largest_black_cluster = None
        cropped_image = cleaned_image.copy()
        save_image(cropped_image, f"{output_dir}/cropped_image.png")
        save_image(cropped_image, f"{output_base2}/cropped_image.png")

    # First pass: remove white noise
    current_image = cropped_image.copy()
    for iteration in range(max_iterations):
        denoised_image, changes_made = remove_noise_white(current_image, THRESHOLD)
        if not changes_made:
            break
        current_image = denoised_image
    save_image(current_image, f"{output_dir}/denoised_image.png")
    save_image(current_image, f"{output_base2}/denoised_image.png")
    # Invert and convert boundary black pixels
    binary_inverted_current_image = np.bitwise_not(current_image)
    save_image(binary_inverted_current_image, f"{output_dir}/denoised_inverted_image.png")
    save_image(binary_inverted_current_image, f"{output_base2}/denoised_inverted_image.png")
    final_image = convert_black_to_white(binary_inverted_current_image)
    save_image(final_image, f"{output_dir}/final_image.png")
    save_image(final_image, f"{output_base2}/final_image.png")

    # Second pass: remove black noise
    current_image_second_pass = final_image.copy()
    for iteration in range(max_iterations):
        denoised_image, changes_made = remove_noise_black(current_image_second_pass, THRESHOLD)
        if not changes_made:
            break
        current_image_second_pass = denoised_image
    final_image_second_pass = current_image_second_pass
    save_image(final_image_second_pass, f"{output_dir}/final_image_second_pass.png")
    save_image(final_image_second_pass, f"{output_base2}/final_image_second_pass.png")

    # Analyze and merge clusters
    clusters, average_size = analyze_black_pixel_clusters(final_image_second_pass)
    clusters = merge_vertical_clusters(clusters)
    clusters = merge_contained_clusters(clusters, containment_threshold=0.7)
    clusters = merge_close_horizontal_clusters(clusters)

    # Statistics
    largest_size, smallest_size, median_size, largest_cluster, cluster_sizes = analyze_cluster_statistics(clusters)
    character_widths, modified_z_scores, raw_deviations, median_width, mad = calculate_character_widths_and_z_scores(clusters)

    # Final character extraction
    final_clusters, char_index = process_final_clusters(
        clusters, modified_z_scores, raw_deviations,
        final_image_second_pass, output_dir, image_no,output_base2
    )

    # Visualizations
    plot_image_processing_results(
        binary_inverted, cleaned_image, cropped_image, current_image,
        binary_inverted_current_image, final_image, final_image_second_pass,
        black_clusters, largest_black_cluster, clusters, final_clusters, width
    )
    plot_labeled_image(final_image_second_pass, final_clusters)

    # Build result for backend/frontend
    result = {
        "image_no": image_no,
        "char_count": len(final_clusters),
        "average_cluster_size": average_size,
        "final_image_path": f"{output_dir}/final_image_second_pass.png",
        "cluster_data": [
            {
                "char_index": cluster['char_index'],
                "bounding_box": cluster['bounding_box'],
                "is_sub_character": cluster['is_sub_character'],
                "width": cluster['bounding_box'][2] - cluster['bounding_box'][0] + 1
            } for cluster in final_clusters
        ],
    }

    return result
