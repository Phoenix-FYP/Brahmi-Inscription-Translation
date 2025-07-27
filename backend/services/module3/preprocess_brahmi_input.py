from utils import normalize_str, split_graphemes

def preprocess_input(raw_text: str):
    """
    Cleans and splits input into grapheme clusters.

    Args:
        raw_text (str): Noisy or unsegmented input text.

    Returns:
        cleaned_text (str): Normalized and cleaned text.
        graphemes (List[str]): List of grapheme clusters.
    """
    cleaned_text = normalize_str(raw_text)
    graphemes = split_graphemes(cleaned_text)
    return cleaned_text, graphemes


# === Optional CLI Debugging Mode ===
if __name__ == "__main__":
    import sys

    # Test input or CLI argument
    example_input = sys.argv[1] if len(sys.argv) > 1 else "අතතනඅතනශය්"

    cleaned, graphemes = preprocess_input(example_input)

    print("🧼 Cleaned Input Text:", cleaned)
    print("🔤 Grapheme Clusters :", graphemes)
