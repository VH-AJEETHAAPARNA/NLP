# Expt.No:7
# Aim: Construct N-Gram and Hidden Markov Model (HMM) language models
# and compare their ability to capture word sequences in tweet data.

import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist
from nltk.tag import hmm
from nltk.corpus import treebank
import matplotlib.pyplot as plt
import warnings

# Suppress runtime warnings from HMM
warnings.filterwarnings("ignore")

# Download required datasets
nltk.download('punkt')
nltk.download('treebank')

# -----------------------------
# Input Tweet
# -----------------------------
tweet = input("Enter a tweet: ")

# Tokenization
tokens = nltk.word_tokenize(tweet.lower())

print("\nTokens:")
print(tokens)

# -----------------------------
# N-Gram Language Model
# -----------------------------
print("\n========== N-GRAM MODEL ==========")

# Unigrams
unigrams = list(ngrams(tokens, 1))
print("\nUnigrams:")
print(unigrams)

# Bigrams
bigrams = list(ngrams(tokens, 2))
print("\nBigrams:")
print(bigrams)

# Trigrams
trigrams = list(ngrams(tokens, 3))
print("\nTrigrams:")
print(trigrams)

# Word Frequency
fd = FreqDist(tokens)

print("\nWord Frequencies:")
for word, freq in fd.items():
    print(word, ":", freq)

# -----------------------------
# Hidden Markov Model (HMM)
# -----------------------------
print("\n========== HMM MODEL ==========")

# Train HMM using Treebank corpus
train_data = treebank.tagged_sents()[:3000]

trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# Predict POS tags
tagged_sentence = hmm_tagger.tag(tokens)

print("\nHMM POS Tagging:")
for word, tag in tagged_sentence:
    print(word, "->", tag)

# -----------------------------
# Comparison
# -----------------------------
print("\n========== COMPARISON ==========")

print("N-Gram Model")
print("- Learns word sequences.")
print("- Predicts the next word based on previous words.")
print("- Used for text generation and language modeling.")

print("\nHMM Model")
print("- Predicts Part-of-Speech (POS) tags.")
print("- Uses transition and emission probabilities.")
print("- Used for sequence labeling tasks.")

# -----------------------------
# Save Results as Image
# -----------------------------
final_output = []
final_output.append("Tokens:\n" + str(tokens))
final_output.append("\nUnigrams:\n" + str(unigrams))
final_output.append("\nBigrams:\n" + str(bigrams))
final_output.append("\nTrigrams:\n" + str(trigrams))
final_output.append("\nWord Frequencies:\n" + str(dict(fd)))
final_output.append("\nHMM POS Tagging:\n" + str(tagged_sentence))
final_output.append("\nComparison:\nN-Gram vs HMM models")

report_text = "\n".join(final_output)

plt.figure(figsize=(10, 8))
plt.text(0.01, 0.99, report_text, fontsize=9, va='top', wrap=True)
plt.axis('off')
plt.savefig("../Outputs/exp7_output.png", bbox_inches='tight')
plt.close()

