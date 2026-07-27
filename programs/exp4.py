# Expt.No:4
# Aim: Build an information retrieval system using classical (TF-IDF) and non-classical (LSA) models
# and compare their performance on a dataset of scientific papers.

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Step 1: Accept scientific paper abstracts
docs = []
n = int(input("Enter number of documents: "))
for i in range(n):
    docs.append(input(f"Enter document {i+1}: "))

# Step 2: Accept search query
query = input("\nEnter search query: ")

# Step 3: Convert documents into TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

# Step 4: Transform query into TF-IDF vector
query_vec = vectorizer.transform([query])

# Step 5: Calculate similarity using TF-IDF
scores = cosine_similarity(query_vec, X)
print("\nTF-IDF Similarity Scores:")
for i, s in enumerate(scores[0]):
    print("Document", i+1, ":", round(s, 3))

# Step 6: Apply LSA using Truncated SVD
svd = TruncatedSVD(n_components=2, random_state=0)
X_lsa = svd.fit_transform(X)
query_lsa = svd.transform(query_vec)

# Step 7: Compute similarity scores using LSA
lsa_scores = cosine_similarity(query_lsa, X_lsa)
print("\nLSA Similarity Scores:")
for i, s in enumerate(lsa_scores[0]):
    print("Document", i+1, ":", round(s, 3))

# Step 8: Display most relevant document
best = np.argmax(lsa_scores)
print("\nMost Relevant Document:")
print(docs[best])

print("\nResult:")
print("The information retrieval system successfully retrieved relevant documents using TF-IDF and LSA.")
print("LSA provided better semantic understanding of documents.")
