"""
src/ml/scorer.py
Modèle ML supervisé pour scorer la compatibilité CV / offre.
Features : similarité cosinus + overlap compétences + features textuelles.
Tracking MLflow intégré.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

load_dotenv()

MODELS_DIR = os.getenv("MODELS_DIR", "./models")
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")


# ------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------

def compute_skill_overlap(cv_skills: list, offre_text: str) -> dict:
    """
    Calculer le chevauchement de compétences entre CV et offre.
    Retourne plusieurs features numériques.
    """
    offre_lower = offre_text.lower()
    cv_set = set([s.lower() for s in cv_skills])

    matched = []
    for skill in cv_set:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, offre_lower):
            matched.append(skill)

    n_cv      = len(cv_set)
    n_matched = len(matched)

    return {
        "skill_overlap_count": n_matched,
        "skill_overlap_ratio": n_matched / max(n_cv, 1),
        "skill_overlap_log":   np.log1p(n_matched),
    }


def compute_text_features(cv_text: str, offre_text: str) -> dict:
    """Features textuelles simples CV vs offre."""
    cv_words    = set(cv_text.lower().split())
    offre_words = set(offre_text.lower().split())

    intersection = cv_words & offre_words
    union        = cv_words | offre_words

    jaccard = len(intersection) / max(len(union), 1)
    offre_len_norm = min(len(offre_text) / 2000, 1.0)

    return {
        "jaccard_similarity": jaccard,
        "offre_length_norm":  offre_len_norm,
        "cv_words_in_offre":  len(intersection) / max(len(cv_words), 1),
    }


def build_features(
    cv_data: dict,
    offres_df: pd.DataFrame,
    cv_embedding: np.ndarray,
    offres_embeddings: np.ndarray,
) -> pd.DataFrame:
    """
    Construire la matrice de features pour toutes les offres.

    Features :
    - cosine_similarity       : similarité sémantique embedding
    - skill_overlap_count     : nb compétences CV dans l'offre
    - skill_overlap_ratio     : ratio compétences matchées
    - skill_overlap_log       : log(overlap) pour réduire l'effet levier
    - jaccard_similarity      : overlap mots CV/offre
    - offre_length_norm       : longueur normalisée de l'offre
    - cv_words_in_offre       : % mots CV présents dans l'offre
    """
    rows = []

    for i, (_, offre) in enumerate(offres_df.iterrows()):
        offre_text = f"{offre.get('intitule','')} {offre.get('description','')} {offre.get('competences','')}"

        # Similarité cosinus (embeddings normalisés)
        cos_sim = float(np.dot(cv_embedding, offres_embeddings[i]))

        # Overlap compétences
        skill_feats = compute_skill_overlap(cv_data["competences"], offre_text)

        # Features textuelles
        text_feats = compute_text_features(cv_data["enriched_text"], offre_text)

        row = {"cosine_similarity": cos_sim}
        row.update(skill_feats)
        row.update(text_feats)
        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Génération des labels (pseudo-labels pour entraînement)
# ------------------------------------------------------------------

def generate_pseudo_labels(features_df: pd.DataFrame) -> np.ndarray:
    """
    Générer des pseudo-labels pour l'entraînement supervisé.
    Combinaison pondérée des features les plus fiables.
    En prod, ces labels seraient remplacés par des annotations humaines.
    """
    labels = (
        0.50 * features_df["cosine_similarity"] +
        0.30 * features_df["skill_overlap_ratio"] +
        0.20 * features_df["jaccard_similarity"]
    )
    # Normaliser entre 0 et 1
    labels = (labels - labels.min()) / (labels.max() - labels.min() + 1e-8)
    return labels.values


# ------------------------------------------------------------------
# Entraînement avec MLflow
# ------------------------------------------------------------------

def train_scorer(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    experiment_name: str = "jobfit_scoring",
) -> tuple:
    """
    Entraîner et comparer plusieurs modèles avec tracking MLflow.
    Retourne le meilleur modèle et ses métriques.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment_name)

    X = features_df.values
    y = labels

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0)),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(n_estimators=100, random_state=42)),
        ]),
        "gradient_boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ]),
    }

    results = {}
    best_score = -np.inf
    best_model = None
    best_name  = ""

    for name, pipeline in models.items():
        with mlflow.start_run(run_name=name):
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring="r2")
            r2_mean   = cv_scores.mean()
            r2_std    = cv_scores.std()

            # Entraîner sur tout le dataset
            pipeline.fit(X, y)
            y_pred = pipeline.predict(X)
            rmse   = np.sqrt(mean_squared_error(y, y_pred))

            # Logger dans MLflow
            mlflow.log_param("model_type",  name)
            mlflow.log_param("n_features",  X.shape[1])
            mlflow.log_param("n_samples",   X.shape[0])
            mlflow.log_metric("r2_cv_mean", r2_mean)
            mlflow.log_metric("r2_cv_std",  r2_std)
            mlflow.log_metric("rmse_train", rmse)
            mlflow.log_metric("r2_train",   r2_score(y, y_pred))
            mlflow.sklearn.log_model(pipeline, f"model_{name}")

            results[name] = {
                "r2_cv_mean": round(r2_mean, 4),
                "r2_cv_std":  round(r2_std,  4),
                "rmse":       round(rmse,     4),
            }

            print(f"  {name:25s} R² CV: {r2_mean:.4f} ± {r2_std:.4f}  RMSE: {rmse:.4f}")

            if r2_mean > best_score:
                best_score = r2_mean
                best_model = pipeline
                best_name  = name

    print(f"\nMeilleur modèle : {best_name} (R²={best_score:.4f})")

    # Sauvegarder le meilleur modèle
    os.makedirs(MODELS_DIR, exist_ok=True)
    import joblib
    model_path = f"{MODELS_DIR}/best_scorer.pkl"
    joblib.dump(best_model, model_path)
    print(f"Modèle sauvegardé : {model_path}")

    return best_model, results


# ------------------------------------------------------------------
# Scoring final
# ------------------------------------------------------------------

def score_offres(
    model,
    features_df: pd.DataFrame,
    offres_df: pd.DataFrame,
) -> pd.DataFrame:
    """Appliquer le modèle et retourner les offres scorées."""
    X = features_df.values
    scores = model.predict(X)
    scores = np.clip(scores, 0, 1)

    df = offres_df.copy()
    df["ml_score"]     = scores
    df["ml_score_pct"] = (scores * 100).round(1)

    return df.sort_values("ml_score", ascending=False)


# ------------------------------------------------------------------
# Usage direct
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import sqlite3
    import joblib

    sys.path.append("../nlp")
    sys.path.append("../api")

    from cv_parser   import parse_cv
    from embeddings  import EmbeddingEngine

    DATA_DIR = os.getenv("DATA_PROCESSED", "../../data/processed")
    CV_PATH  = "../../data/cv/sample_cv.pdf"

    # Charger CV
    cv_data      = parse_cv(CV_PATH)
    engine       = EmbeddingEngine()
    cv_embedding = engine.embed_text(cv_data["enriched_text"])

    # Charger offres et embeddings
    conn         = sqlite3.connect(f"{DATA_DIR}/jobfit.db")
    df_offres    = pd.read_sql("SELECT * FROM offres", conn)
    conn.close()

    offres_embeddings = np.load(f"{DATA_DIR}/offres_embeddings.npy")

    print(f"Offres : {len(df_offres)} | Embeddings : {offres_embeddings.shape}")

    # Features
    print("\nConstruction des features...")
    features_df = build_features(cv_data, df_offres, cv_embedding, offres_embeddings)
    print(features_df.describe().round(3))

    # Labels
    labels = generate_pseudo_labels(features_df)

    # Entraînement
    print("\nEntraînement des modèles...")
    best_model, results = train_scorer(features_df, labels)

    # Scoring final
    df_scored = score_offres(best_model, features_df, df_offres)

    print("\n--- TOP 10 OFFRES (score ML) ---")
    cols = ["intitule", "entreprise", "lieu", "type_contrat", "ml_score_pct"]
    print(df_scored[cols].head(10).to_string(index=False))

    df_scored.to_csv(f"{DATA_DIR}/offres_ml_scored.csv", index=False)
    print(f"\nSauvegardé : {DATA_DIR}/offres_ml_scored.csv")
