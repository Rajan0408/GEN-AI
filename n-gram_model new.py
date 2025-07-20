# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 08:06:17 2025

@author: Rajan
"""

import nltk
from nltk.corpus import brown
from collections import defaultdict, Counter
import random

# Download Brown corpus
nltk.download('brown')
nltk.download('punkt')

# Step 1: Load and preprocess the corpus
sentences = brown.sents()
sentences = [[word.lower() for word in sentence] for sentence in sentences]

# Step 2: Build trigrams
n = 3
trigrams = []

for sentence in sentences:
    if len(sentence) >= n:
        for i in range(len(sentence) - n + 1):
            trigram = tuple(sentence[i:i+n])
            trigrams.append(trigram)

# Step 3: Count trigram frequencies
model = defaultdict(Counter)
for w1, w2, w3 in trigrams:
    model[(w1, w2)][w3] += 1

# Step 4: Predict next word
def predict_next(word1, word2):
    next_words = model[(word1, word2)]
    if not next_words:
        return None
    return next_words.most_common(1)[0][0]

# Step 5: Generate text
def generate_text(start_words, num_words=20):
    w1, w2 = start_words
    generated = [w1, w2]
    
    for _ in range(num_words):
        next_word = predict_next(w1, w2)
        if not next_word:
            break
        generated.append(next_word)
        w1, w2 = w2, next_word
    
    return ' '.join(generated)

# Example Usage
print(generate_text(("the", "president"), num_words=30))
