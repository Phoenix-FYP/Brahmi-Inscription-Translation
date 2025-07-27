# === Stage 6: Sentence-Level Correction with ByT5 ===
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

BYT5_MODEL_DIR = "../model/byt5_brahmi"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

byt5_tokenizer = AutoTokenizer.from_pretrained(BYT5_MODEL_DIR)
byt5_model = AutoModelForSeq2SeqLM.from_pretrained(BYT5_MODEL_DIR).to(DEVICE)
byt5_model.eval()

def correct_with_byt5(input_sentence):
    prompt = f"correct spelling: {input_sentence.strip()}"
    inputs = byt5_tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = byt5_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=32,
            num_beams=4,
            early_stopping=True,
            repetition_penalty=1.5,
            no_repeat_ngram_size=2
        )

    corrected_text = byt5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return corrected_text.strip()

# 🔁 Apply correction
input_for_byt5 =""
corrected_sentence = correct_with_byt5(input_for_byt5)
print("✅ Stage 6 - Corrected Sentence:", corrected_sentence)
