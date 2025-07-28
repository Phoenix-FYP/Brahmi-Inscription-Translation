import regex as regx
from difflib import SequenceMatcher

# === Grapheme-aware splitting ===
GRAPHEME_PATTERN = regx.compile(r'\X', regx.UNICODE)

def split_graphemes(word):
    return GRAPHEME_PATTERN.findall(word)

# === Word-level diffing ===
def word_level_diff(before_words, after_words):
    """
    Returns a list of (index, before_word, after_word) for mismatched words.
    """
    diffs = []
    matcher = SequenceMatcher(None, before_words, after_words)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace' or tag == 'delete' or tag == 'insert':
            max_len = max(i2 - i1, j2 - j1)
            for offset in range(max_len):
                before = before_words[i1 + offset] if i1 + offset < i2 else ''
                after = after_words[j1 + offset] if j1 + offset < j2 else ''
                diffs.append((i1 + offset, before, after))
    return diffs

def print_word_diffs(before_text, after_text):
    """
    Compares two sentences word-by-word and prints differences.
    """
    before_words = before_text.strip().split()
    after_words = after_text.strip().split()

    diffs = word_level_diff(before_words, after_words)

    if not diffs:
        print("✅ No word-level changes.")
        return

    print("\n🔍 Word-Level Differences:")
    for idx, before, after in diffs:
        print(f"   • Word {idx + 1}: '{before}' → '{after}'")

# === CLI Test ===
if __name__ == "__main__":
    # Example: best segmented vs ByT5 output
    segmented_text = "පමමක ශිවහ ලෙණෙ ශගශ"
    corrected_text = "පරුමක ශිවහ ලෙණෙ ශක්\u200dය"

    print("📋 Segmented Sentence:", segmented_text)
    print("🛠️  Corrected Sentence :", corrected_text)

    print_word_diffs(segmented_text, corrected_text)
