import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================
# CONFIG
# =========================
TMDB_API_KEY = "5b4b9300ec4573844ccac77f5bc007c4"  # hardcoded for simplicity
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Movie Recommender API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df = None
indices_obj = None
tfidf_matrix = None
tfidf_obj = None
TITLE_TO_IDX = None

# =========================
# MODELS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    poster_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]

# =========================
# HELPERS
# =========================
def _norm_title(t):
    return str(t).strip().lower()


def make_img_url(path):
    return f"{TMDB_IMG_500}{path}" if path else None


async def tmdb_get(path, params):
    params["api_key"] = TMDB_API_KEY
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{TMDB_BASE}{path}", params=params)
        if r.status_code != 200:
            raise HTTPException(status_code=500, detail="TMDB error")
        return r.json()


# =========================
# LOAD PICKLES
# =========================
@app.on_event("startup")
def load_data():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    df = pickle.load(open(DF_PATH, "rb"))
    indices_obj = pickle.load(open(INDICES_PATH, "rb"))
    tfidf_matrix = pickle.load(open(TFIDF_MATRIX_PATH, "rb"))
    tfidf_obj = pickle.load(open(TFIDF_PATH, "rb"))

    TITLE_TO_IDX = {_norm_title(k): int(v) for k, v in indices_obj.items()}


# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {"msg": "API Running"}


@app.get("/tmdb/search")
async def search_movies(query: str):
    return await tmdb_get("/search/movie", {"query": query})


@app.get("/movie/id/{tmdb_id}")
async def movie_details(tmdb_id: int):
    data = await tmdb_get(f"/movie/{tmdb_id}", {})
    return {
        "tmdb_id": data["id"],
        "title": data["title"],
        "overview": data.get("overview"),
        "poster_url": make_img_url(data.get("poster_path")),
        "genres": data.get("genres", [])
    }


@app.get("/recommend/tfidf")
def recommend(title: str):
    idx = TITLE_TO_IDX[_norm_title(title)]
    scores = (tfidf_matrix @ tfidf_matrix[idx].T).toarray().ravel()
    indices = np.argsort(-scores)[1:11]

    return [
        {"title": df.iloc[i]["title"], "score": float(scores[i])}
        for i in indices
    ]