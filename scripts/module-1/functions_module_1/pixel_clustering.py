from collections import deque
import numpy as np

def count_connected_pixels(image, start_y, start_x, visited, target_value):
    """Count connected pixels and return their coordinates."""
    h, w = image.shape
    count = 0
    pixels = []
    queue = deque([(start_y, start_x)])
    visited[start_y, start_x] = True
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    
    while queue:
        y, x = queue.popleft()
        count += 1
        pixels.append((y, x))
        for dy, dx in directions:
            new_y, new_x = y + dy, x + dx
            if (0 <= new_y < h and 0 <= new_x < w and 
                not visited[new_y, new_x] and 
                image[new_y, new_x] == target_value):
                queue.append((new_y, new_x))
                visited[new_y, new_x] = True
    return count, pixels

def analyze_black_pixel_clusters(image):
    """Analyze black pixel clusters in the image."""
    h, w = image.shape
    visited = np.zeros((h, w), dtype=bool)
    clusters = []
    
    for y in range(h):
        for x in range(w):
            if image[y, x] == 0 and not visited[y, x]:
                pixel_count, pixels = count_connected_pixels(image, y, x, visited, 0)
                if pixels:
                    y_coords, x_coords = zip(*pixels)
                    min_y, max_y = min(y_coords), max(y_coords)
                    min_x, max_x = min(x_coords), max(x_coords)
                    clusters.append({
                        'size': pixel_count,
                        'bounding_box': (min_x, min_y, max_x, max_y),
                        'start_pixel': (y, x),
                        'center': ((min_x + max_x) / 2, (min_y + max_y) / 2),
                        'pixels': pixels
                    })
    
    clusters.sort(key=lambda cluster: cluster['bounding_box'][0])
    average_size = sum(cluster['size'] for cluster in clusters) / len(clusters) if clusters else 0
    return clusters, average_size

def merge_vertical_clusters(clusters):
    """Merge vertically aligned clusters based on overlap and proximity."""
    if not clusters:
        return clusters
    merged_clusters = []
    used = [False] * len(clusters)
    OVERLAP_THRESHOLD = 0.7
    avg_height = sum(cluster['bounding_box'][3] - cluster['bounding_box'][1] + 1 for cluster in clusters) / len(clusters) if clusters else 1
    PROXIMITY_THRESHOLD = avg_height * 0.5
    
    for i, cluster in enumerate(clusters):
        if used[i]:
            continue
        min_x, min_y, max_x, max_y = cluster['bounding_box']
        cluster_width = max_x - min_x + 1
        cluster_height = max_y - min_y + 1
        merged_pixels = set(cluster['pixels'])
        merged_size = cluster['size']
        merged_start_pixel = cluster['start_pixel']
        merged_center = [cluster['center'][0], cluster['center'][1]]
        count = 1
        
        best_overlap = 0
        best_j = -1
        best_is_proximity = False
        
        for j, other_cluster in enumerate(clusters):
            if i == j or used[j]:
                continue
            other_min_x, other_min_y, other_max_x, other_max_y = other_cluster['bounding_box']
            other_width = other_max_x - other_min_x + 1
            other_height = other_max_y - other_min_y + 1
            
            overlap_start = max(min_x, other_min_x)
            overlap_end = min(max_x, other_max_x)
            overlap_width = max(0, overlap_end - overlap_start + 1)
            smaller_width = min(cluster_width, other_width)
            containment_ratio = overlap_width / smaller_width if smaller_width > 0 else 0
            
            is_above = other_max_y < min_y or max_y < other_min_y
            vertical_gap = min(abs(min_y - other_max_y), abs(max_y - other_min_y)) if is_above else float('inf')
            
            if containment_ratio >= OVERLAP_THRESHOLD and containment_ratio > best_overlap:
                best_overlap = containment_ratio
                best_j = j
                best_is_proximity = False
            elif is_above and vertical_gap <= PROXIMITY_THRESHOLD and containment_ratio > 0:
                if best_j == -1 or (vertical_gap < PROXIMITY_THRESHOLD and containment_ratio > best_overlap):
                    best_overlap = containment_ratio
                    best_j = j
                    best_is_proximity = True
        
        if best_j != -1:
            other_cluster = clusters[best_j]
            other_min_x, other_min_y, other_max_x, other_max_y = other_cluster['bounding_box']
            merged_pixels.update(other_cluster['pixels'])
            merged_size += other_cluster['size']
            min_x = min(min_x, other_min_x)
            max_x = max(max_x, other_max_x)
            min_y = min(min_y, other_min_y)
            max_y = max(max_y, other_max_y)
            merged_center[0] = (merged_center[0] * count + other_cluster['center'][0]) / (count + 1)
            merged_center[1] = (merged_center[1] * count + other_cluster['center'][1]) / (count + 1)
            count += 1
            used[best_j] = True
            print(f"Merged cluster at ({min_x}, {min_y}) with cluster at ({other_min_x}, {other_min_y}) "
                  f"via {'proximity' if best_is_proximity else 'overlap'} (overlap: {best_overlap:.2f}, vertical gap: {vertical_gap if best_is_proximity else 'N/A'})")
        
        merged_clusters.append({
            'size': merged_size,
            'bounding_box': (min_x, min_y, max_x, max_y),
            'start_pixel': merged_start_pixel,
            'center': (merged_center[0], merged_center[1]),
            'pixels': list(merged_pixels)
        })
        used[i] = True
    
    merged_clusters.sort(key=lambda cluster: cluster['bounding_box'][0])
    return merged_clusters

def merge_contained_clusters(clusters, containment_threshold=0.7):
    """Merge clusters based on bounding box containment."""
    if not clusters:
        return clusters
    merged_clusters = []
    used = [False] * len(clusters)
    
    for i, cluster in enumerate(clusters):
        if used[i]:
            continue
        min_x, min_y, max_x, max_y = cluster['bounding_box']
        cluster_width = max_x - min_x + 1
        cluster_height = max_y - min_y + 1
        cluster_area = cluster_width * cluster_height
        merged_pixels = set(cluster['pixels'])
        merged_size = cluster['size']
        merged_start_pixel = cluster['start_pixel']
        merged_center = [cluster['center'][0], cluster['center'][1]]
        count = 1
        
        for j, other_cluster in enumerate(clusters[i+1:], start=i+1):
            if used[j]:
                continue
            other_min_x, other_min_y, other_max_x, other_max_y = other_cluster['bounding_box']
            other_width = other_max_x - other_min_x + 1
            other_height = other_max_y - other_min_y + 1
            other_area = other_width * other_height
            
            intersect_min_x = max(min_x, other_min_x)
            intersect_max_x = min(max_x, other_max_x)
            intersect_min_y = max(min_y, other_min_y)
            intersect_max_y = min(max_y, other_max_y)
            intersect_width = max(0, intersect_max_x - intersect_min_x + 1)
            intersect_height = max(0, intersect_max_y - intersect_min_y + 1)
            intersect_area = intersect_width * intersect_height
            
            smaller_area = min(cluster_area, other_area)
            containment_ratio = intersect_area / smaller_area if smaller_area > 0 else 0
            
            if containment_ratio >= containment_threshold:
                merged_pixels.update(other_cluster['pixels'])
                merged_size += other_cluster['size']
                min_x = min(min_x, other_min_x)
                max_x = max(max_x, other_max_x)
                min_y = min(min_y, other_min_y)
                max_y = max(max_y, other_max_y)
                merged_center[0] = (merged_center[0] * count + other_cluster['center'][0]) / (count + 1)
                merged_center[1] = (merged_center[1] * count + other_cluster['center'][1]) / (count + 1)
                count += 1
                used[j] = True
                print(f"Merged cluster at ({other_min_x}, {other_min_y}) "
                      f"into cluster at ({min_x}, {min_y}), "
                      f"containment ratio: {containment_ratio:.2f}")
        
        merged_clusters.append({
            'size': merged_size,
            'bounding_box': (min_x, min_y, max_x, max_y),
            'start_pixel': merged_start_pixel,
            'center': (merged_center[0], merged_center[1]),
            'pixels': list(merged_pixels)
        })
        used[i] = True
    
    merged_clusters.sort(key=lambda cluster: cluster['bounding_box'][0])
    return merged_clusters

def merge_close_horizontal_clusters(clusters):
    """Merge horizontally close clusters for fractured characters."""
    if not clusters:
        return clusters
    merged_clusters = []
    used = [False] * len(clusters)
    avg_width = sum(cluster['bounding_box'][2] - cluster['bounding_box'][0] + 1 for cluster in clusters) / len(clusters) if clusters else 1
    HORIZONTAL_GAP_THRESHOLD = avg_width * 0.01

    for i, cluster in enumerate(clusters):
        if used[i]:
            continue
        min_x, min_y, max_x, max_y = cluster['bounding_box']
        merged_pixels = set(cluster['pixels'])
        merged_size = cluster['size']
        merged_start_pixel = cluster['start_pixel']
        merged_center = [cluster['center'][0], cluster['center'][1]]
        count = 1
        
        for j, other_cluster in enumerate(clusters):
            if i == j or used[j]:
                continue
            other_min_x, other_min_y, other_max_x, other_max_y = other_cluster['bounding_box']
            
            is_right = max_x < other_min_x or other_max_x < min_x
            horizontal_gap = min(abs(max_x - other_min_x), abs(other_max_x - min_x)) if is_right else float('inf')
            
            overlap_y_start = max(min_y, other_min_y)
            overlap_y_end = min(max_y, other_max_y)
            overlap_height = max(0, overlap_y_end - overlap_y_start + 1)
            smaller_height = min(max_y - min_y + 1, other_max_y - other_min_y + 1)
            vertical_overlap_ratio = overlap_height / smaller_height if smaller_height > 0 else 0
            
            if is_right and horizontal_gap <= HORIZONTAL_GAP_THRESHOLD and vertical_overlap_ratio >= 0.5:
                merged_pixels.update(other_cluster['pixels'])
                merged_size += other_cluster['size']
                min_x = min(min_x, other_min_x)
                max_x = max(max_x, other_max_x)
                min_y = min(min_y, other_min_y)
                max_y = max(max_y, other_max_y)
                merged_center[0] = (merged_center[0] * count + other_cluster['center'][0]) / (count + 1)
                merged_center[1] = (merged_center[1] * count + other_cluster['center'][1]) / (count + 1)
                count += 1
                used[j] = True
                print(f"Merged horizontally close cluster at ({other_min_x}, {other_min_y}) "
                      f"into cluster at ({min_x}, {min_y}), "
                      f"horizontal gap: {horizontal_gap:.2f}, vertical overlap ratio: {vertical_overlap_ratio:.2f}")
        
        merged_clusters.append({
            'size': merged_size,
            'bounding_box': (min_x, min_y, max_x, max_y),
            'start_pixel': merged_start_pixel,
            'center': (merged_center[0], merged_center[1]),
            'pixels': list(merged_pixels)
        })
        used[i] = True
    
    merged_clusters.sort(key=lambda cluster: cluster['bounding_box'][0])
    return merged_clusters