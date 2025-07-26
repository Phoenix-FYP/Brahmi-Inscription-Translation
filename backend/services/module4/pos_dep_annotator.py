# pos_dep/pos_dep_annotator.py
import os
import torch
import logging
import stanza

def get_stanza_annotations(words, model_dir, pretokenized=True):
    try:
        tokenize_model_path = os.path.join(model_dir, 'si', 'tokenize', 'si_custom_train.pt')
        pos_model_path = os.path.join(model_dir, 'si_custom_nocharlm_tagger.pt')
        lemma_model_path = os.path.join(model_dir, 'si_custom_nocharlm_lemmatizer.pt')
        depparse_model_path = os.path.join(model_dir, 'si_custom_nocharlm_parser.pt')
        pretrain_path = os.path.join(model_dir, 'si_custom_pretrain.pt')

        nlp = stanza.Pipeline(
            lang='si',
            processors='tokenize,pos,lemma,depparse',
            tokenize_pretokenized=pretokenized,
            dir=model_dir,
            package='si_custom',
            tokenize_model_path=tokenize_model_path,
            lemma_model_path=lemma_model_path,
            pos_model_path=pos_model_path,
            pos_pretrain_path=pretrain_path,
            depparse_model_path=depparse_model_path,
            depparse_pretrain_path=pretrain_path,
            use_gpu=torch.cuda.is_available(),
            use_cache=False
        )

        input_data = [words] if pretokenized else ' '.join(words)
        doc = nlp(input_data)

        tokens, pos_tags, xpos_tags, deps = [], [], [], []
        for sent in doc.sentences:
            for word in sent.words:
                tokens.append(word.text)
                pos_tags.append(word.upos)
                xpos_tags.append(word.xpos or '_')
                deps.append(word.deprel or '_')

        return tokens, pos_tags, xpos_tags, deps

    except Exception as e:
        logging.error(f"Stanza failed: {e}")
        return None, None, None, None
