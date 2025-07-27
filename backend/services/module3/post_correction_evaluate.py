from Levenshtein import distance as levenshtein_distance
from ngram_score import load_ngram_model, score_sentence_ngram
from utils import normalize_str, split_graphemes, grapheme_distance

# === Core post-correction evaluation ===
def post_correction_evaluation(corrected_sentence, original_input, bigram_counts, trigram_counts, dictionary, verbose=True):
    result = {}

    # Stage 7: N-gram score
    ngram_score = score_sentence_ngram(corrected_sentence, bigram_counts, trigram_counts)
    result['ngram_score'] = round(ngram_score, 4)

    # Stage 8: Dictionary OOV check + grapheme-aware fuzzy match
    words = corrected_sentence.strip().split()
    oov_words = []
    fuzzy_matches = {}

    original_words = original_input.strip().split()
    corrected_words = corrected_sentence.strip().split()

    for i, word in enumerate(corrected_words):
        norm_word = normalize_str(word)
        true_word = original_words[i] if i < len(original_words) else "<unknown>"

        if norm_word not in dictionary:
            oov_words.append(word)
            sorted_matches = sorted(dictionary, key=lambda w: grapheme_distance(norm_word, w))
            fuzzy_matches[word] = {
                "true": true_word,
                "suggestions": sorted_matches[:5]
            }

    result['oov_words'] = oov_words
    result['fuzzy_matches'] = fuzzy_matches
    result['num_oov'] = len(oov_words)

    # Stage 9: Grapheme-aware fuzzy accuracy & edit distance
    original_norm = normalize_str(original_input)
    corrected_norm = normalize_str(corrected_sentence)

    original_graphemes = split_graphemes(original_norm)
    corrected_graphemes = split_graphemes(corrected_norm)

    edit_dist = levenshtein_distance(original_graphemes, corrected_graphemes)
    max_len = max(len(original_graphemes), 1)
    accuracy = 1.0 - (edit_dist / max_len)

    result['fuzzy_accuracy'] = round(accuracy, 4)
    result['edit_distance'] = edit_dist
    result['needs_review'] = accuracy < 1.0 or len(oov_words) > 0

    if verbose:
        print("\n📌 Post-Correction Evaluation:")
        print(f"   ➤ N-gram Score       : {result['ngram_score']}")
        print(f"   ➤ OOV Words          : {oov_words}")
        print(f"   ➤ Fuzzy Accuracy     : {accuracy * 100:.2f}%")
        print(f"   ➤ Edit Distance      : {edit_dist}")
        print(f"   ➤ ❗ Needs Review     : {result['needs_review']}")
        for oov, match in fuzzy_matches.items():
            suggestions = ", ".join(match["suggestions"])
            print(f"     - {oov} → True: {match['true']}, Suggestions: {suggestions}")

    return result

# === CLI Test ===
if __name__ == "__main__":
    print("✅ N-gram model loading...")
    bigrams, trigrams = load_ngram_model("../model/ngram_model")

    dict_path = "../data/dictionary/brahmi_dictionary.csv"
    with open(dict_path, "r", encoding="utf-8") as f:
        word_dict = set(normalize_str(line) for line in f if line.strip())

    # === Sample input (simulate pre/post correction) ===
    original_input = "පමමක ශිවහ ලෙණෙ ශගශ"
    corrected_sentence = "පරුමක ශිවහ ලෙණෙ ශක්\u200dය"

    # Run evaluation
    post_correction_evaluation(
        corrected_sentence=corrected_sentence,
        original_input=original_input,
        bigram_counts=bigrams,
        trigram_counts=trigrams,
        dictionary=word_dict
    )
