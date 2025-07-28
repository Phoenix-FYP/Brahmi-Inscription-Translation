from .utils import normalize_str, split_graphemes, count_visual_characters, preprocess_graphemes

# === Forward Greedy Segmentation ===
def greedy_segment_forward(chars, dictionary):
    i = 0
    segments = []

    while i < len(chars):
        match = ''
        for j in range(len(chars), i, -1):
            candidate = ''.join(chars[i:j])
            if candidate in dictionary:
                match = candidate
                segments.append((match, list(range(i, j)), False))
                i = j
                break

        if not match:
            unmatched_start = i
            while i < len(chars):
                next_match = False
                for j in range(len(chars), i + 1, -1):
                    if ''.join(chars[i:j]) in dictionary:
                        next_match = True
                        break
                if next_match:
                    break
                i += 1

            cluster_chars = chars[unmatched_start:i]
            cluster_indices = list(range(unmatched_start, i))
            cluster_string = ''.join(cluster_chars)
            segments.append((cluster_string, cluster_indices, True))

    return segments

# === Backward Greedy Segmentation ===
def greedy_segment_backward(chars, dictionary):
    i = len(chars)
    segments = []

    while i > 0:
        match = ''
        for j in range(0, i):
            candidate = ''.join(chars[j:i])
            if candidate in dictionary:
                match = candidate
                segments.append((match, list(range(j, i)), False))
                i = j
                break

        if not match:
            unmatched_end = i
            while i > 0:
                next_match = False
                for j in range(0, i - 1):
                    if ''.join(chars[j:i]) in dictionary:
                        next_match = True
                        break
                if next_match:
                    break
                i -= 1

            cluster_chars = chars[i:unmatched_end]
            cluster_indices = list(range(i, unmatched_end))
            cluster_string = ''.join(cluster_chars)
            segments.append((cluster_string, cluster_indices, True))

    segments.reverse()
    return segments

# === Main Entry Point ===
def run_bi_directional_greedy_segmentation(raw_text, dictionary):
    graphemes = preprocess_graphemes(raw_text)

    seg_fwd = greedy_segment_forward(graphemes, dictionary)
    seg_bwd = greedy_segment_backward(graphemes, dictionary)

    return {
        "graphemes": graphemes,
        "forward": {
            "segments": seg_fwd,
            "words": [w for w, _, _ in seg_fwd],
            "unmatched": [w for w, _, bad in seg_fwd if bad],
            "segmented_text": " ".join(w for w, _, _ in seg_fwd)
        },
        "backward": {
            "segments": seg_bwd,
            "words": [w for w, _, _ in seg_bwd],
            "unmatched": [w for w, _, bad in seg_bwd if bad],
            "segmented_text": " ".join(w for w, _, _ in seg_bwd)
        }
    }

# # === CLI Test ===
# if __name__ == "__main__":
#     csv_path = "../data/dictionary/brahmi_dictionary.csv"
#     with open(csv_path, "r", encoding="utf-8") as f:
#         sinhala_dict = set(normalize_str(line) for line in f if line.strip())

#     raw_input = "පමමකශිවහලෙණෙශගශ"
#     cleaned_input = normalize_str(raw_input)  # ✅ Normalize before passing
#     result = run_bi_directional_greedy_segmentation(cleaned_input, sinhala_dict)

#     print("\n🔤 Graphemes:")
#     print(result["graphemes"])

#     print("\n➡️ Forward (LTR) Segmentation:")
#     print("   Text:", result["forward"]["segmented_text"])
#     for word, indices, is_unmatched in result["forward"]["segments"]:
#         print(f"   {'❌' if is_unmatched else '✅'} {word} → {indices}")

#     print("\n⬅️ Backward (RTL) Segmentation:")
#     print("   Text:", result["backward"]["segmented_text"])
#     for word, indices, is_unmatched in result["backward"]["segments"]:
#         print(f"   {'❌' if is_unmatched else '✅'} {word} → {indices}")
