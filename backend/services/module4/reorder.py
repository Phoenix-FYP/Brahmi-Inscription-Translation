import torch

def reorder_sentence(sentence, model, tokenizer, device, get_annotations):
    words = sentence.split()
    tokens, pos_tags, xpos_tags, deps = get_annotations(words)
    if tokens is None:
        return None, None, None, None, None

    input_text = f"{' '.join(tokens)} | POS: {' '.join(pos_tags)} | XPOS: {' '.join(xpos_tags)} | Dep: {' '.join(deps)}"
    full_input = f"input: {input_text} → output:"
    inputs = tokenizer(full_input, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
    reordered = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return tokens, pos_tags, xpos_tags, deps, reordered
