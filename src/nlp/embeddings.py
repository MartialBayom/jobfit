"""
src/nlp/embeddings.py
Génération des embeddings sémantiques pour CV et offres d'emploi.
Utilise sentence-transformers avec un modèle multilingue FR/EN.
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDINGS_PATH = os.getenv("DATA_PROCESSED", "./data/processed")


class EmbeddingEngine:
    """Moteur d'embeddings pour CV et offres."""

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"Chargement du modèle : {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Modèle chargé.")

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> np.ndarray:
        """Générer l'embedding d'un texte."""
        return self.model.encode(text, normalize_embeddings=True)

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Générer les embeddings d'une liste de textes."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    # ------------------------------------------------------------------
    # Préparer le texte d'une offre pour l'embedding
    # ------------------------------------------------------------------

    def prepare_offre_text(self, offre: dict | pd.Series) -> str:
        """
        Construire un texte enrichi depuis une offre pour l'embedding.
        Concatène intitulé + description + compétences.
        """
        parts = []
        if offre.get("intitule"):
            parts.append(str(offre["intitule"]))
        if offre.get("description"):
            # Limiter à 500 chars pour équilibrer avec le CV
            parts.append(str(offre["description"])[:500])
        if offre.get("competences"):
            parts.append(f"Compétences requises : {offre['competences']}")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Calcul de similarité cosinus
    # ------------------------------------------------------------------

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Similarité cosinus entre deux vecteurs normalisés."""
        return float(np.dot(vec1, vec2))

    def score_cv_vs_offre(self, cv_embedding: np.ndarray, offre_text: str) -> float:
        """
        Scorer un CV contre une offre.
        Retourne un score entre 0 et 1.
        """
        offre_embedding = self.embed_text(offre_text)
        return self.cosine_similarity(cv_embedding, offre_embedding)

    def score_cv_vs_offres(
        self,
        cv_embedding: np.ndarray,
        offres_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Scorer un CV contre toutes les offres du DataFrame.
        Ajoute une colonne 'score_similarite'.
        """
        df = offres_df.copy()

        # Préparer les textes des offres
        textes = [
            self.prepare_offre_text(row)
            for _, row in df.iterrows()
        ]

        # Embeddings en batch
        print(f"Calcul des embeddings pour {len(textes)} offres...")
        offres_embeddings = self.embed_texts(textes)

        # Scores cosinus
        scores = [
            self.cosine_similarity(cv_embedding, offre_emb)
            for offre_emb in offres_embeddings
        ]

        df["score_similarite"] = scores
        df["score_pct"] = (df["score_similarite"] * 100).round(1)

        return df.sort_values("score_similarite", ascending=False)

    # ------------------------------------------------------------------
    # Sauvegarde / chargement des embeddings offres
    # ------------------------------------------------------------------

    def save_offres_embeddings(self, embeddings: np.ndarray, path: str = None):
        """Sauvegarder les embeddings des offres."""
        path = path or f"{EMBEDDINGS_PATH}/offres_embeddings.npy"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, embeddings)
        print(f"Embeddings sauvegardés : {path} ({embeddings.shape})")

    def load_offres_embeddings(self, path: str = None) -> np.ndarray:
        """Charger les embeddings des offres."""
        path = path or f"{EMBEDDINGS_PATH}/offres_embeddings.npy"
        embeddings = np.load(path)
        print(f"Embeddings chargés : {path} ({embeddings.shape})")
        return embeddings


# ------------------------------------------------------------------
# Usage direct
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.append("../api")

    from cv_parser import parse_cv
    from database import load_offres

    # Parser le CV
    cv_path = sys.argv[1] if len(sys.argv) > 1 else "../../data/cv/mon_cv.pdf"
    cv_data = parse_cv(cv_path)

    # Charger les offres
    df_offres = load_offres()
    print(f"\n{len(df_offres)} offres chargées.")

    # Embeddings
    engine = EmbeddingEngine()
    cv_embedding = engine.embed_text(cv_data["enriched_text"])
    print(f"CV embedding shape : {cv_embedding.shape}")

    # Scorer les offres
    df_scored = engine.score_cv_vs_offres(cv_embedding, df_offres)

    print("\n--- TOP 10 OFFRES LES PLUS COMPATIBLES ---")
    cols = ["intitule", "entreprise", "lieu", "type_contrat", "score_pct"]
    print(df_scored[cols].head(10).to_string(index=False))

    # Sauvegarder
    offres_embeddings = engine.embed_texts(
        [engine.prepare_offre_text(row) for _, row in df_offres.iterrows()]
    )
    engine.save_offres_embeddings(offres_embeddings)

    df_scored.to_csv(f"{EMBEDDINGS_PATH}/offres_scored.csv", index=False)
    print(f"\nRésultats sauvegardés dans data/processed/offres_scored.csv")
