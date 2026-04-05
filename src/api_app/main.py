"""
src/api_app/main.py
API FastAPI JobFit — endpoints de scoring et conseils candidature.
"""

import os
import sys
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tempfile
import sqlite3
from dotenv import load_dotenv
import json

load_dotenv()

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR   = os.getenv("DATA_PROCESSED", os.path.join(BASE_DIR, "data/processed"))
MODELS_DIR = os.getenv("MODELS_DIR",     os.path.join(BASE_DIR, "models"))
CHROMA_DIR = os.getenv("CHROMA_DIR",     os.path.join(BASE_DIR, "data/chroma"))

sys.path.insert(0, os.path.join(BASE_DIR, "src/nlp"))
sys.path.insert(0, os.path.join(BASE_DIR, "src/ml"))
sys.path.insert(0, os.path.join(BASE_DIR, "src/rag"))

from cv_parser  import parse_cv
from embeddings import EmbeddingEngine
from scorer     import build_features, score_offres
from chatbot    import JobFitChatbot, get_missing_skills

import joblib

# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title="JobFit API",
    description="Matcher CV et offres d'emploi — scoring NLP + conseils RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Chargement des modèles au démarrage
# ------------------------------------------------------------------

engine       = None
scorer_model = None
df_offres    = None
offres_embs  = None


@app.on_event("startup")
async def startup():
    global engine, scorer_model, df_offres, offres_embs

    print("Chargement des modèles...")
    engine       = EmbeddingEngine()
    scorer_model = joblib.load(os.path.join(MODELS_DIR, "best_scorer.pkl"))

    conn      = sqlite3.connect(os.path.join(DATA_DIR, "jobfit.db"))
    df_offres = pd.read_sql("SELECT * FROM offres", conn)
    conn.close()

    offres_embs = np.load(os.path.join(DATA_DIR, "offres_embeddings.npy"))
    print(f"API prête — {len(df_offres)} offres chargées.")


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------

class OffreTexte(BaseModel):
    intitule:    str
    description: str
    entreprise:  Optional[str] = ""
    lieu:        Optional[str] = ""
    type_contrat: Optional[str] = ""
    competences: Optional[str] = ""


class ScoreRequest(BaseModel):
    offre: OffreTexte
    top_n: Optional[int] = 10


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "JobFit API v1.0", "status": "ok"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "offres_en_base": len(df_offres) if df_offres is not None else 0,
    }


@app.post("/score/cv-vs-offres")
async def score_cv_vs_offres(cv_file: UploadFile = File(...), top_n: int = 10):
    """
    Scorer un CV (PDF) contre toutes les offres en base.
    Retourne le top N des offres les plus compatibles.
    """
    if not cv_file.filename.endswith(".pdf"):
        raise HTTPException(400, "Le fichier doit être un PDF.")

    # Sauvegarder le PDF temporairement
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await cv_file.read())
        tmp_path = tmp.name

    try:
        cv_data      = parse_cv(tmp_path)
        cv_embedding = engine.embed_text(cv_data["enriched_text"])

        features_df  = build_features(cv_data, df_offres, cv_embedding, offres_embs)
        df_scored    = score_offres(scorer_model, features_df, df_offres)

        top = df_scored.head(top_n)[
            ["id", "intitule", "entreprise", "lieu", "type_contrat", "ml_score_pct"]
        ].to_dict(orient="records")

        return {
            "cv_titre":       cv_data["titre"],
            "nb_competences": cv_data["nb_competences"],
            "top_offres":     top,
        }
    finally:
        os.unlink(tmp_path)


@app.post("/score/offre-texte")
async def score_offre_texte(cv_file: UploadFile = File(...), offre_json: str = Form(...)):
    """Scorer un CV contre une offre copiée-collée."""
    offre_data = json.loads(offre_json)
    offre = OffreTexte(**offre_data)

    if not cv_file.filename.endswith(".pdf"):
        raise HTTPException(400, "Le fichier doit être un PDF.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await cv_file.read())
        tmp_path = tmp.name

    try:
        cv_data      = parse_cv(tmp_path)
        cv_embedding = engine.embed_text(cv_data["enriched_text"])

        offre_text   = f"{offre.intitule} {offre.description} {offre.competences}"
        offre_emb    = engine.embed_text(offre_text)
        score        = float(np.dot(cv_embedding, offre_emb)) * 100

        from cv_parser import ALL_SKILLS
        all_skills   = [s for s, _ in ALL_SKILLS]
        missing      = get_missing_skills(cv_data["competences"], offre_text, all_skills)

        chatbot  = JobFitChatbot(cv_data)
        advice   = chatbot.analyze_offre(offre.dict(), score)

        return {
            "score":            round(score, 1),
            "cv_competences":   cv_data["competences"],
            "missing_skills":   missing,
            "advice":           advice,
        }
    finally:
        os.unlink(tmp_path)