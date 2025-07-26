# utils/postprocess.py

def post_process_gender(sentence, xpos_tags):
    output = sentence.split()
    for i, (word, xpos) in enumerate(zip(output, xpos_tags)):
        if xpos == 'TITLE-FEM':
            if word.endswith('ග'):
                output[i] = word + 'ා'
            elif word.endswith('ල්'):
                output[i] = word[:-1] + 'ලී'
    return ' '.join(output)
