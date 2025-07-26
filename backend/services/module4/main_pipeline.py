# main_pipeline.py
import os
from morph_completion.morph_complete import load_byt5, morph_complete
from pos_dep.pos_dep_annotator import get_stanza_annotations
from reordering.reorder import load_mt5, reorder_sentence
from utils.postprocess import post_process_gender
from utils.save_output import save_output

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
MT5_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'mt5_reorder')
BYT5_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'byt5_sinhala', 'byt_base_old')
STANZA_MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_FILE = os.path.join(BASE_DIR, 'reorder_output.txt')

def main():
    input_sentence = "වෙළෙන්දා සුමන ලෙණ"
    words = input_sentence.split()

    # POS & Dependency
    tokens, pos_tags, xpos_tags, deps = get_stanza_annotations(words, STANZA_MODEL_DIR)
    if tokens is None:
        print("Annotation failed.")
        return

    # Reordering
    mt5_model, mt5_tokenizer, mt5_device = load_mt5(MT5_MODEL_DIR)
    reordered = reorder_sentence(tokens, pos_tags, xpos_tags, deps, mt5_model, mt5_tokenizer, mt5_device)

    # Gender correction
    gender_corrected = post_process_gender(reordered, xpos_tags)

    # Morphological Completion
    byt5_model, byt5_tokenizer, byt5_device = load_byt5(BYT5_MODEL_DIR)
    morph_completed = morph_complete(gender_corrected, byt5_model, byt5_tokenizer, byt5_device)

    # Output
    print(f"Final Output: {morph_completed}")
    save_output(OUTPUT_FILE, input_sentence, tokens, pos_tags, xpos_tags, deps, reordered, gender_corrected, morph_completed)

if __name__ == "__main__":
    main()
