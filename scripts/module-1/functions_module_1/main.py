import numpy as np
import cv2 as cv
from preprocessing import preprocess_image, visualize_preprocessing_steps
from image_io import save_image, create_output_directory
from pixel_clustering import analyze_black_pixel_clusters, merge_vertical_clusters, merge_contained_clusters, merge_close_horizontal_clusters
from noise_removal import remove_noise_white, remove_noise_black, convert_black_to_white
from image_cropping import create_irregular_crop
from clutser_analysis import analyze_cluster_statistics, calculate_character_widths_and_z_scores, process_final_clusters
from visualization import plot_image_processing_results, plot_labeled_image

def main():
    # Configuration
    image_no = 12
    output_dir = "./cropped_final_characters"
    input_path = f"./data/module-1/images/{image_no}.jpg"
    THRESHOLD = 2000
    max_iterations = 10
    
    # Initialize output directory
    create_output_directory(output_dir)
    
    # Preprocess image
    original_image, binary_inverted, intermediate_images = preprocess_image(input_path, output_dir=output_dir, bias=20, kernel_size=(1, 1), dilation_iterations=1)
    
    # Visualize preprocessing steps
    visualize_preprocessing_steps(original_image, intermediate_images)
    image = cv.imread("./cropped_final_characters/cleaned_image_otsu_adjusted.png")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    binary_inverted = cv.bitwise_not(gray)
    # Save the initial dilated image (equivalent to binary_inverted in original code)
    save_image(binary_inverted, "./images/dilated.png")
    
    # Get image dimensions
    height, width = binary_inverted.shape
    
    # Analyze black pixel clusters
    black_clusters, average_size = analyze_black_pixel_clusters(binary_inverted)
    
    # Crop largest black cluster
    cleaned_image = binary_inverted
    save_image(cleaned_image, "./images/cleaned_image.png")
    
    if black_clusters:
        largest_black_cluster = max(black_clusters, key=lambda cluster: cluster['size'])
        cropped_image = create_irregular_crop(cleaned_image, largest_black_cluster['pixels'])
        print("Largest black cluster found")
        save_image(cropped_image, "./images/cropped_image.png")
    else:
        print("\nNo black pixel clusters found in cleaned_image.")
        largest_black_cluster = None
        cropped_image = cleaned_image.copy()
        save_image(cropped_image, "./images/cropped_iamge.png")
    
    # First pass: Noise removal (white pixels)
    current_image = cropped_image.copy()
    iteration = 0
    while iteration < max_iterations:
        print(f"\nFirst Pass - Iteration {iteration + 1}")
        denoised_image, changes_made = remove_noise_white(current_image, THRESHOLD)
        if not changes_made:
            print("No more noise detected in first pass, stopping iterations")
            break
        current_image = denoised_image
        iteration += 1
    save_image(current_image, "./images/denoised_image.png")
    
    # Invert the denoised image
    binary_inverted_current_image = np.bitwise_not(current_image)
    save_image(binary_inverted_current_image, "./images/denoised_inverted_image.png")
    
    # Convert black to white at boundaries
    final_image = convert_black_to_white(binary_inverted_current_image)
    save_image(final_image, "./images/final_image.png")
    
    # Second pass: Noise removal (black pixels)
    current_image_second_pass = final_image.copy()
    iteration = 0
    while iteration < max_iterations:
        print(f"\nSecond Pass - Iteration {iteration + 1}")
        denoised_image_second_pass, changes_made = remove_noise_black(current_image_second_pass, THRESHOLD)
        if not changes_made:
            print("No more noise detected in second pass, stopping iterations")
            break
        current_image_second_pass = denoised_image_second_pass
        iteration += 1
    final_image_second_pass = current_image_second_pass
    save_image(final_image_second_pass, "./images/final_image_second_pass.png")
    
    # Analyze and merge clusters
    clusters, average_size = analyze_black_pixel_clusters(final_image_second_pass)
    clusters = merge_vertical_clusters(clusters)
    clusters = merge_contained_clusters(clusters, containment_threshold=0.7)
    clusters = merge_close_horizontal_clusters(clusters)
    
    # Analyze cluster statistics
    largest_size, smallest_size, median_size, largest_cluster, cluster_sizes = analyze_cluster_statistics(clusters)
    
    # Calculate character widths and Z-scores
    character_widths, modified_z_scores, raw_deviations, median_width, mad = calculate_character_widths_and_z_scores(clusters)
    
    # Process final clusters and save character images
    final_clusters, char_index = process_final_clusters(clusters, modified_z_scores, raw_deviations, final_image_second_pass, output_dir, image_no)
    
    # Print final cluster details
    print("\nCharacter Details (Left to Right):")
    for cluster in final_clusters:
        size = cluster['size']
        min_x, min_y, max_x, max_y = cluster['bounding_box']
        start_y, start_x = cluster['start_pixel']
        char_width = max_x - min_x + 1
        char_index = cluster['char_index']
        is_sub_character = cluster['is_sub_character']
        idx = clusters.index(cluster) if cluster in clusters else 0
        modified_z_score = modified_z_scores[idx] if idx < len(modified_z_scores) else 0
        raw_deviation = raw_deviations[idx] if idx < len(raw_deviations) else 0
        print(f"Character {char_index}:")
        print(f"  Size: {size} pixels")
        print(f"  Bounding Box: (min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y})")
        print(f"  Starting Pixel: ({start_y}, {start_x})")
        print(f"  Width: {char_width} pixels")
        print(f"  Modified Z-Score: {modified_z_score:.2f}")
        print(f"  Raw Deviation: {raw_deviation:.2f}")
        print(f"  Is Sub-Character: {is_sub_character}")
        if is_sub_character:
            print(f"  ALERT: Character {char_index} is a divided sub-character")
    print(f"\nTotal number of character clusters: {len(final_clusters)}")
    print(f"Average size of character clusters: {average_size:.2f} pixels")
    
    # Visualize results
    plot_image_processing_results(
        binary_inverted, cleaned_image, cropped_image, current_image,
        binary_inverted_current_image, final_image, final_image_second_pass,
        black_clusters, largest_black_cluster, clusters, final_clusters, width
    )
    plot_labeled_image(final_image_second_pass, final_clusters)

if __name__ == "__main__":
    main()