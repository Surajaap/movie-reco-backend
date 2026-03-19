import os
import pickle
from typing import Optional, List
 
import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
 
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"
 
app = FastAPI(title="Movie Recommender API")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# ── Load PKL files ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
df = None
indices = None
tfidf_matrix = None
 
@app.on_event("startup")
def load_pickles():
    global df, indices, tfidf_matrix
    with open(os.path.join(BASE_DIR, "df.pkl"), "rb") as f:
        df = pickle.load(f)
    with open(os.path.join(BASE_DIR, "indices.pkl"), "rb") as f:
        indices = pickle.load(f)
    with open(os.path.join(BASE_DIR, "tfidf_matrix.pkl"), "rb") as f:
        tfidf_matrix = pickle.load(f)
    print("✅ PKL files loaded!")
 
 
# ── TMDB Helper ──
async def tmdb_get(path: str, params: dict) -> dict:
    params["api_key"] = TMDB_API_KEY
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(f"{TMDB_BASE}{path}", params=params)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"TMDB error: {r.status_code}")
    return r.json()
 
 
def make_poster(path: Optional[str]) -> Optional[str]:
    return f"{TMDB_IMG}{path}" if path else None
 
 
# ── Routes ──
 
@app.get("/health")
def health():
    return {"status": "ok"}
 
 
@app.get("/home")
async def home(category: str = Query("trending"), limit: int = Query(20)):
    """Get home feed: trending / popular / top_rated / now_playing / upcoming"""
    if category == "trending":
        data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
    elif category in {"popular", "top_rated", "now_playing", "upcoming"}:
        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
    else:
        raise HTTPException(status_code=400, detail="Invalid category")
 
    results = data.get("results", [])[:limit]
    return [
        {
            "tmdb_id": m["id"],
            "title": m.get("title", ""),
            "poster_url": make_poster(m.get("poster_path")),
            "vote_average": m.get("vote_average"),
            "release_date": m.get("release_date", ""),
        }
        for m in results
    ]
 
 
@app.get("/movie/{tmdb_id}")
async def movie_details(tmdb_id: int):
    """Get movie details by TMDB ID"""
    data = await tmdb_get(f"/movie/{tmdb_id}", {"language": "en-US"})
    return {
        "tmdb_id": data["id"],
        "title": data.get("title", ""),
        "overview": data.get("overview", ""),
        "poster_url": make_poster(data.get("poster_path")),
        "backdrop_url": make_poster(data.get("backdrop_path")),
        "release_date": data.get("release_date", ""),
        "genres": data.get("genres", []),
        "vote_average": data.get("vote_average"),
    }
 
 
@app.get("/search")
async def search(query: str = Query(..., min_length=1)):
    """Search movies on TMDB"""
    data = await tmdb_get("/search/movie", {
        "query": query,
        "language": "en-US",
        "include_adult": "false",
    })
    results = data.get("results", [])[:10]
    return [
        {
            "tmdb_id": m["id"],
            "title": m.get("title", ""),
            "poster_url": make_poster(m.get("poster_path")),
            "release_date": m.get("release_date", ""),
        }
        for m in results
    ]
 
 
@app.get("/recommend/{tmdb_id}")
async def recommend(tmdb_id: int, top_n: int = Query(10)):
    """Get TF-IDF recommendations for a movie"""
    # Get title from TMDB
    data = await tmdb_get(f"/movie/{tmdb_id}", {"language": "en-US"})
    title = data.get("title", "").strip()
 
    # Normalize lookup
    title_lower = title.lower()
    idx_map = {}
    for k, v in indices.items():
        idx_map[str(k).strip().lower()] = int(v)
 
    if title_lower not in idx_map:
        return []
 
    idx = idx_map[title_lower]
    scores = (tfidf_matrix @ tfidf_matrix[idx].T).toarray().ravel()
    order = np.argsort(-scores)
 
    recs = []
    for i in order:
        if int(i) == idx:
            continue
        rec_title = str(df.iloc[int(i)]["title"])
 
        # Fetch poster from TMDB
        try:
            search_data = await tmdb_get("/search/movie", {
                "query": rec_title,
                "language": "en-US",
            })
            results = search_data.get("results", [])
            if results:
                m = results[0]
                recs.append({
                    "tmdb_id": m["id"],
                    "title": m.get("title", rec_title),
                    "poster_url": make_poster(m.get("poster_path")),
                    "release_date": m.get("release_date", ""),
                })
        except Exception:
            recs.append({"tmdb_id": None, "title": rec_title, "poster_url": None})
 
        if len(recs) >= top_n:
            break
 
    return recs