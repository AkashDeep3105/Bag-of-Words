# Bag of Words (BoW) - NLP Project

## Overview
This project demonstrates the implementation of the Bag of Words (BoW) model in Natural Language Processing (NLP). The BoW model is a fundamental method used to convert textual data into numerical representations, enabling machine learning algorithms to process and analyze natural language.

## Project Objective
To transform a set of textual sentences into numerical feature vectors using the Bag of Words technique, allowing further text-based analysis and modeling.

## Tools & Libraries Used
- Python
- `scikit-learn` (`CountVectorizer`)

## Key Concepts
- **Tokenization**: Breaking text into individual words or tokens.
- **Vocabulary Creation**: Collecting all unique words across the text corpus.
- **Vectorization**: Representing each sentence as a vector of word counts.

## How It Works
1. **Input**: A list of text documents/sentences.
2. **CountVectorizer**: Automatically tokenizes the text, builds the vocabulary, and creates a document-term matrix.
3. **Output**: A sparse matrix representing the frequency of each word in each sentence.

## Sample Code Snippet
```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "John likes to watch movies. Mary likes movies too.",
    "John also likes to watch football games."
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

## Output
- A list of all unique words (vocabulary)
- A 2D array showing the frequency of each vocabulary word in each document

## Applications
- Text Classification (Spam Detection, Sentiment Analysis)
- Information Retrieval
- Document Clustering
- Recommendation Systems

## Future Enhancements
- Add stopword removal
- Apply stemming or lemmatization
- Integrate TF-IDF weighting
- Use the vectors for machine learning models like Naive Bayes or SVM


