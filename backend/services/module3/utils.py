import regex as re
import unicodedata
from Levenshtein import distance as levenshtein_distance

# === Patterns ===
INVISIBLE_CHAR_PATTERN = re.compile(r'[\u200B\u200C\u200D\u2060\s]')
GRAPHEME_PATTERN = re.compile(r'\X', re.UNICODE)

# === Normalize: Remove invisible chars and apply NFC normalization ===
def normalize_str(text: str) -> str:
    text = unicodedata.normalize("NFC", text.strip())
    return INVISIBLE_CHAR_PATTERN.sub('', text)

# === Grapheme Cluster Split ===
def split_graphemes(text: str) -> list[str]:
    return GRAPHEME_PATTERN.findall(text)

# === Count Visual Characters ===
def count_visual_characters(s):
    return len(split_graphemes(s))

# === Preprocess Grapheme clusters ===
def preprocess_graphemes(text):
    return split_graphemes(normalize_str(text))

def grapheme_distance(a, b):
    return levenshtein_distance(split_graphemes(a), split_graphemes(b))

# === Checking words in dictionary ===
def dictionary_coverage(words, dictionary):
    if not words:
        return 0.0
    return sum(1 for word in words if normalize_str(word) in dictionary) / len(words)

# === Flagging words not in dictionary ===
def flag_oov(words, dictionary):
    return [i for i, word in enumerate(words) if normalize_str(word) not in dictionary]

# === BIES Decoding Helper ===
def decode_bies(graphemes: list[str], tags: list[str]) -> list[str]:
    words = []
    current = []
    for g, tag in zip(graphemes, tags):
        if tag == "B":
            if current:
                words.append("".join(current))
            current = [g]
        elif tag == "I":
            current.append(g)
        elif tag == "E":
            current.append(g)
            words.append("".join(current))
            current = []
        elif tag == "S":
            if current:
                words.append("".join(current))
            words.append(g)
            current = []
    if current:
        words.append("".join(current))
    return words
