# utils/save_output.py
import logging

def save_output(file_path, input_sentence, tokens, pos_tags, xpos_tags, deps, reordered, gender_corrected, morph_completed):
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(f"{'='*50}\n")
            f.write(f"Input Sentence: {input_sentence}\n")
            f.write(f"Tokens: {tokens}\n")
            f.write(f"POS Tags: {pos_tags}\n")
            f.write(f"XPOS Tags: {xpos_tags}\n")
            f.write(f"Dependencies: {deps}\n")
            f.write(f"Reordered Sentence: {reordered}\n")
            f.write(f"Gender-Corrected Sentence: {gender_corrected}\n")
            f.write(f"Morphologically Completed Sentence: {morph_completed}\n")
            f.write(f"{'='*50}\n\n")
    except Exception as e:
        logging.error(f"Failed to save output: {e}")
