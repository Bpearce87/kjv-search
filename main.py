import json
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

print("Loading KJV clean data and vector embeddings into memory...")
with open("kjv_clean.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    REFERENCES = data["references"]
    TEXTS = data["texts"]

VECTORS = np.load("kjv_vectors.npy")
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("Model and vectors loaded successfully!")

class SearchRequest(BaseModel):
    query: str
    mode: str

@app.post("/api/search")
def search_kjv(request: SearchRequest):
    query_vector = MODEL.encode([request.query], normalize_embeddings=True)[0]
    similarities = np.dot(VECTORS, query_vector)
    top_indices = np.argsort(similarities)[::-1][:8]

    results = [
        {
            "reference": REFERENCES[idx],
            "text": TEXTS[idx],
            "score": float(similarities[idx])
        }
        for idx in top_indices
    ]

    return {
        "mode": request.mode,
        "query": request.query,
        "results": results
    }

@app.get("/")
def home():
    return FileResponse("index.html")