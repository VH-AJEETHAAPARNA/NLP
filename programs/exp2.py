import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import treebank
from nltk.tag.hmm import HiddenMarkovModelTrainer

# Download required resources
nltk.download('punkt')
nltk.download('treebank')

# Prepare training data from the Penn Treebank corpus
train_data = treebank.tagged_sents()

# Train HMM tagger
trainer = HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# Get input from user
text = input("Enter a sentence: ")

# Tokenize sentence
tokens = word_tokenize(text)

# Perform POS tagging using HMM
tagged_words = hmm_tagger.tag(tokens)

# Display tokens
print("\nTokens:")
print(tokens)

# Display POS tags
print("\nPOS Tags:")
for word, tag in tagged_words:
    print(word, "->", tag)

# Simple tag meanings
print("\nTag Meanings:")
print("NN -> Noun")
print("VB -> Verb")
print("JJ -> Adjective")
print("RB -> Adverb")
print("PRP -> Pronoun")
print("DT -> Determiner")

# Count tagged words
print("\nTotal Words:", len(tokens))
