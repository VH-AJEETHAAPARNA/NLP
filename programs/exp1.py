# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.stem import PorterStemmer, WordNetLemmatizer


# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')


# text = input("Enter a sentence or paragraph: ")


# sentences = sent_tokenize(text)
# print("\nSentence Tokens:")
# print(sentences)


# stemmer = PorterStemmer()
# lemmatizer = WordNetLemmatizer()


# for i, sent in enumerate(sentences, 1):
#     words = word_tokenize(sent)
#     print(f"\nWords in sentence {i}:", words)

    
#     stemmed_words = [stemmer.stem(word) for word in words]
#     print("Stemmed:", stemmed_words)

   
#     lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
#     print("Lemmatized:", lemmatized_words)

# print("\nComparison:")
# print("Stemming reduces words to root forms, which may not be meaningful.")
# print("Lemmatization converts words to meaningful base forms.")
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('wordnet')

text = input("Enter a sentence or paragraph: ")

sentences = sent_tokenize(text)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

output_lines = []
output_lines.append("Sentence Tokens:")
output_lines.append(str(sentences))

for i, sent in enumerate(sentences, 1):
    words = word_tokenize(sent)
    output_lines.append(f"\nWords in sentence {i}: {words}")

    stemmed_words = [stemmer.stem(word) for word in words]
    output_lines.append(f"Stemmed: {stemmed_words}")

    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    output_lines.append(f"Lemmatized: {lemmatized_words}")

output_lines.append("\nComparison:")
output_lines.append("Stemming reduces words to root forms, which may not be meaningful.")
output_lines.append("Lemmatization converts words to meaningful base forms.")

# Convert all collected text into one string
final_output = "\n".join(output_lines)

# Render text into an image
plt.figure(figsize=(8, 6))
plt.text(0.01, 0.99, final_output, fontsize=10, va='top', wrap=True)
plt.axis('off')
plt.savefig("../Outputs/exp1_output.png", bbox_inches='tight')
plt.close()

