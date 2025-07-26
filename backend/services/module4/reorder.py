# reordering/reorder.py
import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

def load_mt5(model_dir):
    tokenizer = MT5Tokenizer.from_pretrained(model_dir)
    model = MT5ForConditionalGeneration.from_pretrained(model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer, device

def reorder_sentence(tokens, pos_tags, xpos_tags, deps, model, tokenizer, device):
    input_text = f"reorder: {' '.join(tokens)} | POS: {' '.join(pos_tags)} | XPOS: {' '.join(xpos_tags)} | Dep: {' '.join(deps)}"
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=50).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)

    reordered = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reordered
