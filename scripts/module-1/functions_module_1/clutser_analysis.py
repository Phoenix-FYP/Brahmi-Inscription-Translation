import numpy as np
import os
import cv2 as cv

def analyze_cluster_statistics(clusters):
    """Calculate statistics for black pixel clusters."""
    if not clusters:
        print("\nNo black pixel clusters found.")
        return None, None, None, None, []
    
    cluster_sizes = [cluster['size'] for cluster in clusters]
    largest_size = max(cluster_sizes)
    smallest_size = min(cluster_sizes)
    median_size = np.median(cluster_sizes)
    largest_cluster = max(clusters, key=lambda cluster: cluster['size'])
    
    print("\nBlack Pixel Cluster Statistics:")
    print(f"Largest cluster size: {largest_size} pixels")
    print(f"Smallest cluster size: {smallest_size} pixels")
    print(f"Median cluster size: {median_size:.2f} pixels")
    
    return largest_size, smallest_size, median_size, largest_cluster, cluster_sizes

def calculate_character_widths_and_z_scores(clusters):
    """Calculate character widths and Modified Z-Scores."""
    character_widths = []
    modified_z_scores = []
    raw_deviations = []
    
    if clusters:
        for cluster in clusters:
            min_x, _, max_x, _ = cluster['bounding_box']
            char_width = max_x - min_x + 1
            character_widths.append(char_width)
        
        if character_widths:
            median_width = np.median(character_widths)
            raw_deviations = [width - median_width for width in character_widths]
            absolute_deviations = [abs(dev) for dev in raw_deviations]
            mad = np.median(absolute_deviations) if absolute_deviations else 0
            if mad != 0:
                modified_z_scores = [0.6745 * abs(dev) / mad for dev in raw_deviations]
            else:
                modified_z_scores = [0] * len(character_widths)
                print("Warning: MAD is zero, setting Modified Z-Scores to 0.")
            
            print(f"\nCharacter Width Statistics (Modified Z-Score Method):")
            print(f"Character widths: {character_widths}")
            print(f"Median width: {median_width:.2f} pixels")
            print(f"MAD: {mad:.2f} pixels")
            print(f"Raw Deviations: {[f'{dev:.2f}' for dev in raw_deviations]}")
            print(f"Modified Z-Scores: {[f'{z:.2f}' for z in modified_z_scores]}")
            print(f"Unusual threshold: Modified Z-Score > 3 (split into two for positive deviations only)")
        else:
            print("\nNo character widths to analyze.")
            median_width = mad = None
    else:
        print("\nNo character clusters to analyze.")
        median_width = mad = None
    
    return character_widths, modified_z_scores, raw_deviations, median_width, mad

def process_final_clusters(clusters, modified_z_scores, raw_deviations, image, output_dir, image_no):
    """Process clusters, handle outliers, and save character images."""
    final_clusters = []
    char_index = 1
    
    for filename in os.listdir(output_dir):
        if filename.startswith(f"{image_no}_image_character_") and filename.endswith(".png"):
            file_path = os.path.join(output_dir, filename)
            os.remove(file_path)
            print(f"Removed existing file: {file_path}")
    
    for idx, cluster in enumerate(clusters):
        min_x, min_y, max_x, max_y = cluster['bounding_box']
        char_width = max_x - min_x + 1
        modified_z_score = modified_z_scores[idx] if idx < len(modified_z_scores) else 0
        raw_deviation = raw_deviations[idx] if idx < len(raw_deviations) else 0
        
        if modified_z_score > 3 and raw_deviation > 0:
            print(f"\nDividing high outlier cluster: Width={char_width:.2f}, Modified Z-Score={modified_z_score:.2f}, Raw Deviation={raw_deviation:.2f}")
            print(f"  Bounding Box: (min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y})")
            print(f"  Size: {cluster['size']} pixels")
            
            mid_x = min_x + (max_x - min_x) // 2
            sub_pixels1 = [(y, x) for y, x in cluster['pixels'] if min_x <= x <= mid_x]
            if sub_pixels1:
                sub_size1 = len(sub_pixels1)
                sub_center_x1 = sum(x for y, x in sub_pixels1) / sub_size1
                sub_center_y1 = sum(y for y, x in sub_pixels1) / sub_size1
                final_clusters.append({
                    'size': sub_size1,
                    'bounding_box': (min_x, min_y, mid_x, max_y),
                    'start_pixel': (min_y, min_x),
                    'center': (sub_center_x1, sub_center_y1),
                    'pixels': sub_pixels1,
                    'char_index': char_index,
                    'is_sub_character': True
                })
                char_filename1 = os.path.join(output_dir, f"{image_no}_image_character_{char_index}.png")
                char_image1 = image[min_y:max_y+1, min_x:mid_x+1]
                cv.imwrite(char_filename1, char_image1)
                print(f"Saved sub-character {char_index} to {char_filename1}")
                char_index += 1
            
            sub_pixels2 = [(y, x) for y, x in cluster['pixels'] if mid_x + 1 <= x <= max_x]
            if sub_pixels2:
                sub_size2 = len(sub_pixels2)
                sub_center_x2 = sum(x for y, x in sub_pixels2) / sub_size2
                sub_center_y2 = sum(y for y, x in sub_pixels2) / sub_size2
                final_clusters.append({
                    'size': sub_size2,
                    'bounding_box': (mid_x + 1, min_y, max_x, max_y),
                    'start_pixel': (min_y, mid_x + 1),
                    'center': (sub_center_x2, sub_center_y2),
                    'pixels': sub_pixels2,
                    'char_index': char_index,
                    'is_sub_character': True
                })
                char_filename2 = os.path.join(output_dir, f"{image_no}_image_character_{char_index}.png")
                char_image2 = image[min_y:max_y+1, mid_x+1:max_x+1]
                cv.imwrite(char_filename2, char_image2)
                print(f"Saved sub-character {char_index} to {char_filename2}")
                char_index += 1
        else:
            cluster['char_index'] = char_index
            cluster['is_sub_character'] = False
            final_clusters.append(cluster)
            char_filename = os.path.join(output_dir, f"{image_no}_image_character_{char_index}.png")
            char_image = image[min_y:max_y+1, min_x:max_x+1]
            cv.imwrite(char_filename, char_image)
            print(f"Saved character {char_index} to {char_filename}")
            char_index += 1
    
    return final_clusters, char_index