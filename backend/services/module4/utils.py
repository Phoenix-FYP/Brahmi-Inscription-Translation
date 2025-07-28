import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(CURRENT_DIR, '..', '..', 'results', 'module4', 'final_output.txt')

def save_output(mapped, tokens, pos_tags, xpos_tags, deps, reordered, gendered, morphed, brhami):
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write(f"Brahmi Sentence (Input Words): {' '.join(brhami)}\n")
        f.write(f"Mapped Sinhala Sentence: {mapped}\n")
        f.write(f"POS Tags: {pos_tags}\n")
        f.write(f"XPOS Tags: {xpos_tags}\n")
        f.write(f"Dependencies: {deps}\n")
        f.write(f"Reordered Sentence: {reordered}\n")
        f.write(f"Gender-Corrected Sentence: {gendered}\n")
        f.write(f"Morphologically Completed Sentence: {morphed}\n")
        f.write("="*50 + "\n\n")
