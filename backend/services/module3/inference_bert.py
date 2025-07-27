from transformers import AutoModelForTokenClassification, PreTrainedTokenizerFast
import torch
from preprocess_brahmi_input import preprocess_input
from utils import split_graphemes, decode_bies

# === Config
MODEL_PATH = "../model/sinhala_bert_bies_model/final"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load tokenizer and model
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

def segment_with_bert(text):
    graphemes = split_graphemes(text)
    inputs = tokenizer(graphemes, is_split_into_words=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
        tags = [model.config.id2label[p] for p in predictions[:len(graphemes)]]
    return decode_bies(graphemes, tags)

# === Test
if __name__ == "__main__":
    test_input = "පමමකශිවහලෙණෙශගශ"
    cleaned_input = preprocess_input(test_input)
    segmented = segment_with_bert(test_input)
    print("🧾 Input:", test_input)
    print("🪄 Segmented:", " | ".join(segmented))
