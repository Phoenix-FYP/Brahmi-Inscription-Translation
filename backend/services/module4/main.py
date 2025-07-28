from .dictionary import map_words_to_meanings
from .models import load_model
from .stanza_processing import get_stanza_annotations
from .reorder import reorder_sentence
from .gender import post_process_gender
from .morph import morph_complete
from .utils import save_output

import os

REORDER_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'module-4', 'byt5_reorder')
MORPH_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'module-4', 'byt5_sinhala', 'byt_base_old')

reorder_model, reorder_tokenizer, reorder_device = load_model(REORDER_MODEL_DIR)
morph_model, morph_tokenizer, morph_device = load_model(MORPH_MODEL_DIR)

def run_pipeline(brhami_words):
    input_sentence = map_words_to_meanings(brhami_words)
    if not input_sentence:
        return {"error": "Mapping failed"}

    tokens, pos_tags, xpos_tags, deps, reordered = reorder_sentence(
        input_sentence, reorder_model, reorder_tokenizer, reorder_device, get_stanza_annotations
    )
    if tokens is None:
        return {"error": "Reordering failed"}

    gendered = post_process_gender(reordered, xpos_tags)
    morphed = morph_complete(gendered, morph_model, morph_tokenizer, morph_device)

    save_output(input_sentence, tokens, pos_tags, xpos_tags, deps, reordered, gendered, morphed, brhami_words)

    return {
        "brhami_words": brhami_words,
        "mapped_sentence": input_sentence,
        "reordered": reordered,
        "gender_corrected": gendered,
        "morph_completed": morphed
    }

if __name__ == "__main__":
    brhami_words = ['බමණ', 'උතර', 'පුත', 'ගුතහ', 'ලෙණෙ', 'ශගශ']
    print(run_pipeline(brhami_words))
