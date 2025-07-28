import unicodedata
import re

def clean_unicode_text(text):
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[\u200b\u200c\u200d\u200e\u200f]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
