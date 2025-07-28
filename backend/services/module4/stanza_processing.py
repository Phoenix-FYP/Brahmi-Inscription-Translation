import stanza
import os
import torch
import logging

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STANZA_MODEL_DIR = os.path.join(CURRENT_DIR, '..', '..', '..', 'models', 'module-4')

def get_stanza_annotations(words):
    try:
        tokenize_model_path = os.path.join(STANZA_MODEL_DIR, 'si', 'tokenize', 'si_custom_train.pt')
        pos_model_path = os.path.join(STANZA_MODEL_DIR, 'si_custom_nocharlm_tagger.pt')
        lemma_model_path = os.path.join(STANZA_MODEL_DIR, 'si_custom_nocharlm_lemmatizer.pt')
        depparse_model_path = os.path.join(STANZA_MODEL_DIR, 'si_custom_nocharlm_parser.pt')
        pretrain_path = os.path.join(STANZA_MODEL_DIR, 'si_custom_pretrain.pt')

        nlp = stanza.Pipeline(
            lang='si',
            processors='tokenize,pos,lemma,depparse',
            tokenize_pretokenized=True,
            dir=STANZA_MODEL_DIR,
            package='si_custom',
            tokenize_model_path=tokenize_model_path,
            lemma_model_path=lemma_model_path,
            pos_model_path=pos_model_path,
            pos_pretrain_path=pretrain_path,
            depparse_model_path=depparse_model_path,
            depparse_pretrain_path=pretrain_path,
            download_method=stanza.DownloadMethod.REUSE_RESOURCES,
            use_cache=False,
            verbose=True,
            use_gpu=torch.cuda.is_available()
        )

        doc = nlp([words])
        tokens, pos_tags, xpos_tags, deps = [], [], [], []

        for sent in doc.sentences:
            for word in sent.words:
                tokens.append(word.text)
                pos_tags.append(word.upos)
                xpos_tags.append(word.xpos or '_')
                deps.append(word.deprel or '_')

        return tokens, pos_tags, xpos_tags, deps
    except Exception as e:
        logging.error(f"Stanza failed: {str(e)}")
        return None, None, None, None
