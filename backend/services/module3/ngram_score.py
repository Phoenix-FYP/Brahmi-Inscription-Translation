import math
import pickle
import re

def tokenize(text):
    return re.findall(r'\S+', text)

def load_ngram_model(load_dir="../model/ngram_model"):
    with open(f"{load_dir}/bigram_counts.pkl", "rb") as f:
        bigram_counts = pickle.load(f)
    with open(f"{load_dir}/trigram_counts.pkl", "rb") as f:
        trigram_counts = pickle.load(f)
    print(f"✅ N-gram model loaded from '{load_dir}/'")
    return bigram_counts, trigram_counts

def score_sentence_ngram(sentence, bigram_counts, trigram_counts, k=1):
    tokens = ['<s>', '<s>'] + tokenize(sentence) + ['</s>']
    score = 0.0

    for i in range(len(tokens) - 2):
        trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
        bigram = (tokens[i], tokens[i + 1])
        trigram_count = trigram_counts.get(trigram, 0)
        bigram_count = bigram_counts.get(bigram, 0)

        prob = (trigram_count + k) / (bigram_count + k * len(trigram_counts))
        score += math.log(prob)

    return score
#
# # === CLI Test ===
# if __name__ == "__main__":
#     bigrams, trigrams = load_ngram_model()
#
#     test_sentences = [
#         "පමමක ශිවහ ලෙණෙ ශගශ",
#         "ප ම ම ක ශිවහ ලෙණෙ ශගශ"
#     ]
#
#     print("\n📊 Sentence Fluency Scores:")
#     for sent in test_sentences:
#         score = score_sentence_ngram(sent, bigrams, trigrams)
#         print(f"→ '{sent}': {score:.4f}")