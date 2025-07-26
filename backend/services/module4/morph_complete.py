# morph_completion/morph_complete.py
import os
import torch
from transformers import T5ForConditionalGeneration, ByT5Tokenizer

def load_byt5(model_dir):
    model_dir = os.path.abspath(model_dir)
    tokenizer = ByT5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir, use_safetensors=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer, device

def morph_complete(sentence, model, tokenizer, device):
    input_text = f"input: {sentence} → output:"
    inputs = tokenizer(input_text, return_tensors='pt', max_length=256, truncation=True).to(device)

    outputs = model.generate(
        input_ids=inputs['input_ids'],
        max_length=256,
        num_beams=4,
        temperature=0.7,
        top_k=50,
        do_sample=True,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)