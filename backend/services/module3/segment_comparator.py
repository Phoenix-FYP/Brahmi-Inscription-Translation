import unicodedata
from ngram_score import load_ngram_model, score_sentence_ngram
from utils import normalize_str, dictionary_coverage, flag_oov

def identify_mismatches(seg1, seg2):
    return [i for i, (a, b) in enumerate(zip(seg1, seg2)) if a != b]

def normalize_scores(score_dict):
    scores = list(score_dict.values())
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return {k: 1.0 for k in score_dict}
    return {
        k: (score_dict[k] - min_s) / (max_s - min_s)
        for k in score_dict
    }

def custom_confidence(score, score_min, score_max, words, dictionary, mismatches):
    if len(mismatches) == 0 and dictionary_coverage(words, dictionary) == 1.0:
        return 1.0

    fluency_score_norm = (score - score_min) / (score_max - score_min + 1e-6)
    dict_coverage = dictionary_coverage(words, dictionary)
    agreement_score = 1.0 if len(mismatches) == 0 else 0.5 if len(mismatches) <= 2 else 0.0
    return (
        0.5 * fluency_score_norm +
        0.3 * dict_coverage +
        0.2 * agreement_score
    )

# === Main Comparison Function ===
def compare_segmentations_and_flag_errors(
    bert_segmentation,
    greedy_ltr_segmentation,
    greedy_rtl_segmentation,
    dictionary,
    bigram_counts, trigram_counts,
    fluency_threshold=-60
):
    def score(sent):
        return score_sentence_ngram(sent, bigram_counts, trigram_counts)

    candidates = {
        "BERT": {"words": bert_segmentation, "sentence": " ".join(bert_segmentation)},
        "LTR":  {"words": greedy_ltr_segmentation, "sentence": " ".join(greedy_ltr_segmentation)},
        "RTL":  {"words": greedy_rtl_segmentation, "sentence": " ".join(greedy_rtl_segmentation)},
    }

    scores = {}
    for method in candidates:
        candidates[method]["score"] = score(candidates[method]["sentence"])
        candidates[method]["oov_indices"] = flag_oov(candidates[method]["words"], dictionary)
        candidates[method]["is_fluent"] = candidates[method]["score"] > fluency_threshold
        scores[method] = candidates[method]["score"]

    score_min = min(scores.values())
    score_max = max(scores.values())
    for method in candidates:
        mismatches = identify_mismatches(candidates[method]["words"], bert_segmentation)
        conf = custom_confidence(
            score=candidates[method]["score"],
            score_min=score_min,
            score_max=score_max,
            words=candidates[method]["words"],
            dictionary=dictionary,
            mismatches=mismatches
        )
        candidates[method]["confidence"] = round(conf, 2)

    mismatches = {
        "BERT_vs_LTR": identify_mismatches(bert_segmentation, greedy_ltr_segmentation),
        "BERT_vs_RTL": identify_mismatches(bert_segmentation, greedy_rtl_segmentation),
        "LTR_vs_RTL": identify_mismatches(greedy_ltr_segmentation, greedy_rtl_segmentation)
    }

    return {
        "candidates": candidates,
        "mismatches": mismatches
    }

# === CLI test ===
if __name__ == "__main__":
    # Load N-gram model
    bigrams_loaded, trigrams_loaded = load_ngram_model("../model/ngram_model")

    # Load word-level dictionary
    word_dict_path = "../data/dictionary/brahmi_dictionary.csv"
    with open(word_dict_path, "r", encoding="utf-8") as f:
        word_dict = set(normalize_str(line) for line in f if line.strip())

    # Sample input segmentations
    bert_words = ['පරුමක', 'ශිවහ', 'ලෙණෙ', 'ශගශ']
    ltr_words  = ['පමමක', 'ශිවහ', 'ලෙණෙ', 'ශගශ']
    rtl_words  = ['ප', 'ම', 'ම', 'ක', 'ශිවහ', 'ලෙණෙ', 'ශගශ']

    result = compare_segmentations_and_flag_errors(
        bert_segmentation=bert_words,
        greedy_ltr_segmentation=ltr_words,
        greedy_rtl_segmentation=rtl_words,
        dictionary=word_dict,
        bigram_counts=bigrams_loaded,
        trigram_counts=trigrams_loaded
    )

    print("\n📊 Candidate Evaluation:")
    for method, data in result["candidates"].items():
        print(f"{method}: Score={data['score']:.2f}, Confidence={data['confidence']:.2f}, "
              f"OOV={len(data['oov_indices'])}, Fluent={data['is_fluent']}")

    print("\n⚠️ Mismatches:")
    for k, v in result["mismatches"].items():
        print(f"{k}: indices → {v}")

    # 🏆 Select best candidate
    best_method = max(result["candidates"], key=lambda k: result["candidates"][k]["confidence"])
    best_candidate = result["candidates"][best_method]

    print(f"\n🏅 Best Candidate: {best_method}")
    print("Segmented Sentence:", " | ".join(best_candidate["words"]))

    # ❓ Decision for ByT5
    needs_correction = (
        best_candidate["confidence"] < 0.9 or
        not best_candidate["is_fluent"] or
        len(best_candidate["oov_indices"]) > 0
    )
    print("→ Needs Correction?:", needs_correction)

    if needs_correction:
        input_for_byt5 = " ".join(best_candidate["words"])
        print("🚀 Passing to ByT5 model:", input_for_byt5)
        # → Call your inference API or model here
