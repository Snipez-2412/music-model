from fastapi import FastAPI, APIRouter
import logging
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import unicodedata
import regex as re
import pandas as pd
from plsa_model import predict_song_topic
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model-api")
app = FastAPI()

llm_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
llm_model = SentenceTransformer(llm_model_name)

df = pd.read_pickle("songs_with_embeddings.pkl")
df_lyrics = pd.read_pickle("lyrics-embeddings.pkl")

def preprocess_lyrics_multilingual_refined(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[^\s\p{L}\p{N}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_llm_embedding(text):
    return llm_model.encode([text])[0]

# user input based recommendation
class InputRequest(BaseModel):
    description: str
    top_k: int = 5

@app.post("/model-api/user-input")
def recommend(req: InputRequest):
    embedding = get_llm_embedding(req.description)
    similarities = cosine_similarity([embedding], np.stack(df['embedding']))[0]
    top_indices = similarities.argsort()[::-1][:req.top_k]
    results = df.iloc[top_indices][['Song', 'Genre']].copy()
    results['Similarity'] = similarities[top_indices]
    response = []
    for _, row in results.iterrows():
        response.append({
            "model": "embedding",
            "song": row["Song"],
            "genre": row["Genre"],
            "score": row["Similarity"]
        })

    return response

# song name based recommendation
class SongNameRequest(BaseModel):
    description: str
    top_k: int = 3

@app.post("/model-api/song-name")
async def recommend_by_song(request: SongNameRequest):
    song_name = request.description
    top_k = request.top_k

    # Check if the song exists in the DataFrame
    song_row = df[df['Song'].str.lower() == song_name.lower()]
    if song_row.empty:
        # Return a 404 status code with the error message
        return JSONResponse(
            status_code=404,
            content={"error": f"Song '{song_name}' not found."}
        )

    try:
        # Generate embedding for the input song
        input_embedding = song_row['embedding'].values[0].reshape(1, -1)
        similarities = cosine_similarity(input_embedding, np.stack(df['embedding']))[0]

        # Exclude the input song itself from the recommendations
        song_index = song_row.index[0]
        similarities[song_index] = -1

        # Get the top-k similar songs
        top_indices = similarities.argsort()[::-1][:top_k]
        recommendations = df.iloc[top_indices][['Song', 'Genre']].copy()
        recommendations['Similarity'] = similarities[top_indices]

        # Normalize the results
        response = []
        for _, row in recommendations.iterrows():
            response.append({
                "model": "embedding",
                "song": row["Song"],
                "genre": row["Genre"],
                "score": row["Similarity"]
            })

        return response

    except Exception as e:
        # Handle unexpected errors
        logger.error("Error in song-name recommendation: %s", str(e))
        return JSONResponse(
            status_code=500,
            content={"error": "An unexpected error occurred while processing the request."}
        )


# embedding +  pLSA based recommendation
class CombinedLyricsRequest(BaseModel):
    lyrics: str
    top_k: int = 5

@app.post("/model-api/lyrics")
async def recommend_combined_lyrics(request: CombinedLyricsRequest):
    lyrics = request.lyrics
    top_k = request.top_k

    logger.info("Received lyrics for combined recommendation (first 100 chars): %s", lyrics[:100])
    logger.info("Requested top_k: %d", top_k)

    if not lyrics.strip():
        return {"error": "Lyrics cannot be empty"}

    response = []

    # Embedding recommendation
    try:
        cleaned_lyrics = preprocess_lyrics_multilingual_refined(lyrics)
        embedding = llm_model.encode([cleaned_lyrics])
        similarities = cosine_similarity(embedding, np.stack(df_lyrics["embedding"].values))[0]
        top_indices = similarities.argsort()[::-1][:top_k]
        lyrics_results = df_lyrics.iloc[top_indices][['Song', 'Genre']].copy()
        lyrics_results['similarity_score'] = similarities[top_indices]

        # Normalize embedding results
        for _, row in lyrics_results.iterrows():
            response.append({
                "model": "embedding",
                "song": row["Song"],
                "genre": row["Genre"],
                "score": row["similarity_score"]
            })

        logger.info("Embedding-based recommendation returned %d results", len(lyrics_results))
    except Exception as e:
        logger.error("Error in embedding recommendation: %s", str(e))

    # pLSA recommendation
    try:
        top_topic, top_topic_prob, related_songs = predict_song_topic(lyrics, top_n=top_k)

        if top_topic is not None:
            # Normalize pLSA results
            for song_data in related_songs:
                response.append({
                    "model": "plsa",
                    "song": song_data["song"],
                    "genre": song_data["genre"],
                    "score": song_data["probability"]
                })

            logger.info("pLSA-based recommendation returned %d related songs", len(related_songs))
        else:
            logger.warn("pLSA-based recommendation returned no results")
    except Exception as e:
        logger.error("Error in pLSA-based recommendation: %s", str(e))

    return response