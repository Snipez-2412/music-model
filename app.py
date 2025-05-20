from fastapi import FastAPI, APIRouter
import logging
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import unicodedata
import regex as re
import pandas as pd

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
    return results.to_dict(orient="records")

# song name based recommendation
class SongNameRequest(BaseModel):
    description: str
    top_k: int = 3

@app.post("/model-api/song-name")
async def recommend_by_song(request: SongNameRequest):
    song_name = request.description
    top_k = request.top_k
    song_row = df[df['Song'].str.lower() == song_name.lower()]
    if song_row.empty:
        return {"error": f"Không tìm thấy bài hát '{song_name}'."}
    input_embedding = song_row['embedding'].values[0].reshape(1, -1)
    similarities = cosine_similarity(input_embedding, np.stack(df['embedding']))[0]
    song_index = song_row.index[0]
    similarities[song_index] = -1
    top_indices = similarities.argsort()[::-1][:top_k]
    recommendations = df.iloc[top_indices][['Song', 'Genre']].copy()
    recommendations['Similarity'] = similarities[top_indices]
    return recommendations.to_dict(orient="records")

# lyrics based recommendation
class CustomLyricsRequest(BaseModel):
    lyrics: str
    top_k: int = 5

@app.post("/model-api/lyrics")
async def recommend_by_lyrics(request: CustomLyricsRequest):
    lyrics = request.lyrics
    top_k = request.top_k

    logger.info("Received lyrics (first 100 chars): %s", lyrics[:100])
    logger.info("Requested top_k: %d", top_k)

    if not lyrics.strip():
        return {"error": "Lyrics cannot be empty"}

    cleaned_lyrics = preprocess_lyrics_multilingual_refined(lyrics)
    embedding = llm_model.encode([cleaned_lyrics])
    similarities = cosine_similarity(embedding, np.stack(df_lyrics["embedding"].values))[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    results = df_lyrics.iloc[top_indices][['Song', 'Genre']].copy()
    results['similarity_score'] = similarities[top_indices]

    logger.info("Returning %d recommendations", len(results))
    return results.to_dict(orient="records")
