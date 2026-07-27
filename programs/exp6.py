import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score

# Download required NLTK resources
nltk.download('punkt')

# Define relation keywords
keywords = ["treats", "reduces", "controls", "helps", "inhibits", "prevents", "associated"]

# Accept biomedical sentence and actual relation label
sentence = input("Enter biomedical sentence: ")
actual = int(input("Actual Relation (1/0): "))

# Convert to lowercase and tokenize
tokens = word_tokenize(sentence.lower())

print("\nTokens:")
print(tokens)

# Rule-based relation prediction
predicted = 0
for word in tokens:
    if word in keywords:
        predicted = 1
        break

print("\nPredicted Relation:", predicted)

# Prepare lists for evaluation
y_true = [actual]
y_pred = [predicted]

# Calculate metrics
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# Display results
print("\nEvaluation Results:")
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Result statement
print("\nResult:")
print("Biomedical relations were successfully identified using a rule-based approach.")
print("The system evaluated performance using Precision, Recall, and F1-Score.")
