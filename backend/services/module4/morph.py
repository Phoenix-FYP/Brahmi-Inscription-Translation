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
