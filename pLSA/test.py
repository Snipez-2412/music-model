import os
import json
from sklearn.feature_extraction.text import CountVectorizer

cleaned_lyrics_path = 'all_cleaned_lyrics.json'
try:
    with open(cleaned_lyrics_path, 'r', encoding='utf-8') as f:
        all_cleaned_lyrics = json.load(f)
    print(f"Successfully loaded {len(all_cleaned_lyrics)} items from all_cleaned_lyrics.json")
except FileNotFoundError:
    print(f"File not found: {cleaned_lyrics_path}")
except json.JSONDecodeError:
    print(f"Error decoding JSON from file: {cleaned_lyrics_path}")

vectorizer = CountVectorizer(max_df=0.97, min_df=3, stop_words='english')
X_corpus = vectorizer.fit_transform(all_cleaned_lyrics)  # Fit the vectorizer on the existing corpus

print(X_corpus.shape)