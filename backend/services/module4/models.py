import os
import torch
from transformers import ByT5Tokenizer, T5ForConditionalGeneration

def load_model(model_dir):
    try:
        from safetensors.torch import load_file
        print("Using safetensors")
    except ImportError:
        print("safetensors not installed")

    tokenizer = ByT5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir, use_safetensors=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer, device
