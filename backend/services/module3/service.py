import os
from .main import run_pipeline as raw_pipeline

# Build safe absolute paths
MODULE3_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MODULE3_DIR, "../../../data/module-3/dictionary")
MODEL_DIR = os.path.join(MODULE3_DIR, "../../../models/module-3/ngram_model")

# Patch the main module paths by monkey-patching globals (only needed once)
from services.module3 import main
main.WORD_DICT_PATH = os.path.join(DATA_DIR, "brahmi_dictionary.csv")
main.NGRAM_MODEL_PATH = MODEL_DIR

# Re-load resources with new paths
with open(main.WORD_DICT_PATH, "r", encoding="utf-8") as f:
    main.brahmi_dict = set(main.normalize_str(line) for line in f if line.strip())

main.bigram_counts, main.trigram_counts = main.load_ngram_model(main.NGRAM_MODEL_PATH)

def run_module3(text: str) -> dict:
    """
    Wrapper function to run Module 3.
    Accepts raw grapheme string output from Module 2.
    Returns segmentation, correction, and ngram evaluation results.
    """
    return raw_pipeline(text)
