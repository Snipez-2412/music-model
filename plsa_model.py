# Models
import re
import os
import numpy as np
import pickle
import nltk
import json
from sklearn.feature_extraction import text
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


stemmer = PorterStemmer()
nltk.download('punkt_tab')
nltk.download('stopwords')


# Load model function
def load_model(filename='plsa_model.pkl'):
    model_path = os.path.join('pLSA', filename)
    with open(model_path, 'rb') as f:
        P_d_z, P_w_z, P_z = pickle.load(f)
    return P_d_z, P_w_z, P_z


def preprocess_lyrics(lyrics):
    # Check if the lyrics are valid (not null and are strings)
    if isinstance(lyrics, str):
        # Remove special characters, numbers, etc.
        lyrics = re.sub(r'[^a-zA-Z\s]', '', lyrics)
        # Convert to lowercase
        lyrics = lyrics.lower()

        # Tokenize the text
        tokens = word_tokenize(lyrics)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        # Apply stemming
        tokens = [stemmer.stem(word) for word in tokens]

        # Join back the processed tokens into a string
        lyrics = ' '.join(tokens)
    else:
        # If not a valid string, return an empty string
        lyrics = ''

    return lyrics


def predict_song_topic(new_lyrics, top_n=3):
    """
    Predict the top topics for a new song based on its lyrics, and find the most related songs in the corpus.
    """
    try:
        # Load the pLSA model
        P_d_z, P_w_z, P_z = load_model()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, []

    # Step 1: Preprocess the new lyrics
    preprocessed_lyrics = preprocess_lyrics(new_lyrics)

    cleaned_lyrics_path = os.path.join('pLSA', 'all_cleaned_lyrics.json')
    try:
        with open(cleaned_lyrics_path, 'r', encoding='utf-8') as f:
            all_cleaned_lyrics = json.load(f)
        print(f"Successfully loaded {len(all_cleaned_lyrics)} items from all_cleaned_lyrics.json")
    except FileNotFoundError:
        print(f"File not found: {cleaned_lyrics_path}")
        return None, None, []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {cleaned_lyrics_path}")
        return None, None, []

    # Step 2: Fit or use the existing vectorizer
    extra_stopwords = ['hmmmmm', 'ah', 'someth', 'caus', 'kany', 'ill', 'wan',
                       'ive', 'want', 'id', 'ayo', 'arent', 'laci', 'steve', 'na',
                       'daniel', 'caesar', 'mayb', 'em', 'oh', 'song', 'lyrics',
                       'chorus', 'kendrick', 'lamar', 'choru', 'ye', 'ooh',
                       'dont', 'kanye', 'vincent', 'aah', 'vers', 'like', 'intro',
                       'hello', 'aaliyah', 'skit', 'hmmmmm ', 'aint', 'im', 'yeah',
                       'yo', 'brent', 'faiyaz', 'mm']

    custom_stop_words = list(text.ENGLISH_STOP_WORDS.union(extra_stopwords))

    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=custom_stop_words)
    X_corpus = vectorizer.fit_transform(all_cleaned_lyrics)  # Fit the vectorizer on the existing corpus

    # Step 3: Vectorize the new lyrics using the fitted vectorizer
    new_X = vectorizer.transform([preprocessed_lyrics]).toarray()

    # Step 4: Calculate the topic distribution for the new lyrics
    P_z_new = np.dot(new_X, P_w_z)  # P(z|d) = X * P(w|z)
    P_z_new /= P_z_new.sum()  # Normalize

    # Step 5: Get the top topic for the new song
    top_index = np.argmax(P_z_new[0])  # Get the index of the highest probability
    top_topic_probability = P_z_new[0][top_index]
    print(f"\nTop Topic for the New Song: Topic {top_index + 1} with Probability {top_topic_probability:.4f}")

    # Step 6: Calculate the topic distribution for all songs in the corpus
    P_z_corpus = np.dot(X_corpus.toarray(), P_w_z)  # P(z|d) = X * P(w|z)

    # Step 6.1: Add a small epsilon value to prevent division by zero
    row_sums = P_z_corpus.sum(axis=1, keepdims=True)
    epsilon = 1e-10
    row_sums[row_sums == 0] = epsilon
    P_z_corpus /= row_sums  # Normalize for each song

    # Step 7: Find the songs with the highest probability for the top topic
    related_song_indices = np.argsort(P_z_corpus[:, top_index])[::-1]  # Sort by highest probability for the top topic

    song_indices_path = os.path.join('pLSA', 'song_indices_with_genre.json')
    try:
        with open(song_indices_path, 'r', encoding='utf-8') as f:
            song_indices = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {song_indices_path}")
        return None, None, []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {song_indices_path}")
        return None, None, []

    # Step 9: Output the related songs with genres
    print("Top Related Songs for the New Song's Top Topic:")
    related_songs = []
    for idx in related_song_indices[:top_n]:
        song_data = song_indices[idx]  # Retrieve song data from the loaded JSON
        song_name, song_genre = song_data  # Extract the name and genre
        song_probability = P_z_corpus[idx, top_index]
        print(f"Song: {song_name} | Genre: {song_genre} | Probability {song_probability:.4f}")
        related_songs.append({
            "song": song_name,
            "genre": song_genre,
            "probability": song_probability
        })

    return top_index + 1, top_topic_probability, related_songs