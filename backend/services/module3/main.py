from .inference_bert import segment_with_bert
from .greedy_segmenter import run_bi_directional_greedy_segmentation
from .segment_comparator import compare_segmentations_and_flag_errors
from .ngram_score import load_ngram_model
from .utils import normalize_str
from .post_correction_evaluate import post_correction_evaluation
import os
import json
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NGRAM_MODEL_PATH = os.path.join(BASE_DIR, "../../../models/module-3/ngram_model")
WORD_DICT_PATH = os.path.join(BASE_DIR, "../../../data/module-3/dictionary/brahmi_dictionary.csv")

# === Load dictionary and N-gram model


with open(WORD_DICT_PATH, "r", encoding="utf-8") as f:
    brahmi_dict = set(normalize_str(line) for line in f if line.strip())

bigram_counts, trigram_counts = load_ngram_model(NGRAM_MODEL_PATH)

# === Temporary Mock ByT5 Correction ===
def mock_byt5_correction(input_text):
    # In reality, this would be a call to your ByT5 model
    # For now, just simulate a plausible correction
    corrections = {
        "පමමක ශිවහ ලෙණෙ ශගශ": "පරුමක ශිවහ ලෙණෙ ශගශ",  # Simulate a known correction
    }
    return corrections.get(input_text, input_text)  # Return input if no mock correction

def run_pipeline(raw_input):
    global corrected_sentence, evaluation_result
    print("Raw Input:", raw_input)

    # Stage 2 - BERT
    bert_words = segment_with_bert(raw_input)
    print("BERT:", bert_words)

    # Stage 3 - Greedy
    greedy_result = run_bi_directional_greedy_segmentation(raw_input, brahmi_dict)
    ltr_words = [w for w, _, _ in greedy_result["forward"]["segments"]]
    rtl_words = [w for w, _, _ in greedy_result["backward"]["segments"]]

    print("LTR:", ltr_words)
    print("RTL:", rtl_words)

    # Stage 4–5 - Scoring & Error Flagging
    comparison = compare_segmentations_and_flag_errors(
        bert_segmentation=bert_words,
        greedy_ltr_segmentation=ltr_words,
        greedy_rtl_segmentation=rtl_words,
        dictionary=brahmi_dict,
        bigram_counts=bigram_counts,
        trigram_counts=trigram_counts
    )

    print("\nCandidate Evaluation:")
    for method, data in comparison["candidates"].items():
        print(f"{method}: Score={data['score']:.2f}, Confidence={data['confidence']:.2f}, OOV={len(data['oov_indices'])}, Fluent={data['is_fluent']}")

    print("\nMismatches:")
    for k, v in comparison["mismatches"].items():
        print(f"{k}: indices → {v}")

    # Select best candidate
    best_method = max(comparison["candidates"], key=lambda k: comparison["candidates"][k]["confidence"])
    best_candidate = comparison["candidates"][best_method]

    print(f"\n Best Candidate: {best_method}")
    print("Segmented Sentence:", " ".join(best_candidate["words"]))

    # Decision for ByT5
    all_same = (
            comparison["mismatches"]["BERT_vs_LTR"] == [] and
            comparison["mismatches"]["BERT_vs_RTL"] == [] and
            comparison["mismatches"]["LTR_vs_RTL"] == []
    )
    all_fluent = all(c["is_fluent"] for c in comparison["candidates"].values())
    all_oov_free = all(len(c["oov_indices"]) == 0 for c in comparison["candidates"].values())

    if all_same and all_fluent and all_oov_free:
        needs_correction = False
    else:
        needs_correction = (
                best_candidate["confidence"] < 0.9 or
                not best_candidate["is_fluent"] or
                len(best_candidate["oov_indices"]) > 0
        )
    print("→ Needs Correction?:", needs_correction)

    if needs_correction:
        input_for_byt5 = " ".join(best_candidate["words"])
        print("Passing to ByT5 model:", input_for_byt5)
        #Call the byt5 Model later
        corrected_sentence = mock_byt5_correction(input_for_byt5)
        print("✅ Corrected Sentence:", corrected_sentence)

        evaluation_result = post_correction_evaluation(
            corrected_sentence=corrected_sentence,
            original_input=input_for_byt5,  # This is the segmented input passed to ByT5
            bigram_counts=bigram_counts,
            trigram_counts=trigram_counts,
            dictionary=brahmi_dict,
            verbose=True
        )

    return {
        "input": raw_input,
        "bert": bert_words,
        "ltr": ltr_words,
        "rtl": rtl_words,
        "result": comparison,
        "best": best_candidate,
        "needs_correction": needs_correction,
        "corrected": corrected_sentence if needs_correction else None,
        "evaluation": evaluation_result if needs_correction else None,
        "fuzzy_matches": evaluation_result.get("fuzzy_matches", {}) if needs_correction else {},
        "oov_words": evaluation_result.get("oov_words", []) if needs_correction else [],
        "fuzzy_accuracy": evaluation_result.get("fuzzy_accuracy") if needs_correction else None,
        "edit_distance": evaluation_result.get("edit_distance") if needs_correction else None,
        "ngram_score": evaluation_result.get("ngram_score") if needs_correction else None
    }

# === Entry point
if __name__ == "__main__":
    sample = "පමමකශිවහලෙණෙශගශ"  # Update this as needed
    result = run_pipeline(sample)

    # === Prepare output directory
    ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../"))
    OUTPUT_DIR = os.path.join(ROOT_DIR, "results", "module3")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === Build output dict
    output_data = {
        "best": result.get("best", {}).get("words", []),
        "corrected": result.get("corrected", "").split() if result.get("corrected") else [],
        "full_result": result
    }

    # === Generate file name
    safe_id = normalize_str(sample)[:10].replace(" ", "_") or "output"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_filename = f"{safe_id}_{timestamp}.json"

    # === Save to file
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nOutput saved to: {output_path}")
