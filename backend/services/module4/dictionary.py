import pandas as pd
import logging
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DICTIONARY_FILE = os.path.join(CURRENT_DIR, '..', '..', '..', 'data', 'module-4', 'dictionary.xlsx')

def map_words_to_meanings(word_list):
    try:
        df = pd.read_excel(DICTIONARY_FILE)
        word_map = dict(zip(df['Brhami Word'], df['Meaning']))
        meanings, missing_words = [], []

        for word in word_list:
            if word in word_map:
                meanings.append(word_map[word])
            else:
                missing_words.append(word)

        if missing_words:
            msg = f"Error: Words not in dictionary: {', '.join(missing_words)}"
            logging.warning(msg)
            with open('error_log.txt', 'a', encoding='utf-8') as f:
                f.write(msg + '\n')
            return None

        return ' '.join(meanings)
    except Exception as e:
        logging.error(f"Dictionary mapping failed: {str(e)}")
        return None
