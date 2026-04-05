"""
src/ml/clustering.py
Clustering des offres d'emploi avec KMeans et HDBSCAN.
Visualisation 2D avec UMAP/TSNE + analyse des clusters.
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")

import mlflow
from dotenv import load_dotenv
load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
DATA_DIR   = os.getenv("DATA_PROCESSED", "./data/processed")


# ------------------------------------------------------------------
# Réduction de dimension pour visualisation
# ------------------------------------------------------------------

def reduce_dimensions(embeddings: np.ndarray, n_components: int = 2, method: str = "pca") -> np.ndarray:
    """
    Réduire les embeddings en 2D pour visualisation.
    method: 'pca' (rapide) ou 'tsne' (plus précis mais lent)
    """
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
        return reducer.fit_transform(embeddings)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        return reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Méthode inconnue : {method}")


# ------------------------------------------------------------------
# Trouver le K optimal (méthode du coude + silhouette)
# ------------------------------------------------------------------

def find_optimal_k(embeddings: np.ndarray, k_range: range = range(2, 12)) -> dict:
    """
    Trouver le nombre optimal de clusters.
    Retourne les inertias et silhouette scores pour chaque K.
    """
    inertias   = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        inertias.append(km.inertia_)
        sil = silhouette_score(embeddings, labels, sample_size=min(500, len(embeddings)))
        silhouettes.append(sil)
        print(f"  K={k:2d} | Inertia: {km.inertia_:,.0f} | Silhouette: {sil:.4f}")

    return {
        "k_range":    list(k_range),
        "inertias":   inertias,
        "silhouettes": silhouettes,
        "best_k":     list(k_range)[np.argmax(silhouettes)],
    }


# ------------------------------------------------------------------
# KMeans clustering
# ------------------------------------------------------------------

def kmeans_clustering(
    embeddings: np.ndarray,
    n_clusters: int,
    experiment_name: str = "jobfit_clustering",
) -> tuple:
    """
    Appliquer KMeans et logger dans MLflow.
    Retourne les labels et le modèle.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"kmeans_k{n_clusters}"):
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)

        sil_score = silhouette_score(embeddings, labels, sample_size=min(500, len(embeddings)))
        db_score  = davies_bouldin_score(embeddings, labels)

        mlflow.log_param("n_clusters",    n_clusters)
        mlflow.log_param("algorithm",     "kmeans")
        mlflow.log_metric("silhouette",   sil_score)
        mlflow.log_metric("davies_bouldin", db_score)
        mlflow.log_metric("inertia",      km.inertia_)

        print(f"KMeans K={n_clusters} | Silhouette: {sil_score:.4f} | Davies-Bouldin: {db_score:.4f}")

    return labels, km


# ------------------------------------------------------------------
# Analyse des clusters
# ------------------------------------------------------------------

def analyze_clusters(
    df_offres: pd.DataFrame,
    labels: np.ndarray,
    n_clusters: int,
) -> pd.DataFrame:
    """
    Analyser le contenu de chaque cluster.
    Retourne un DataFrame avec les stats par cluster.
    """
    df = df_offres.copy()
    df["cluster"] = labels

    cluster_stats = []
    for c in range(n_clusters):
        df_c = df[df["cluster"] == c]

        # Top intitulés
        top_titles = df_c["intitule"].value_counts().head(3).index.tolist()

        # Top compétences
        all_comps = []
        for comp_str in df_c["competences"].dropna():
            all_comps.extend([c.strip() for c in comp_str.split(",") if c.strip()])
        from collections import Counter
        top_comps = [c for c, _ in Counter(all_comps).most_common(5)]

        # Type contrat dominant
        top_contrat = df_c["type_contrat"].mode().iloc[0] if len(df_c) > 0 else "N/A"

        # Top lieu
        top_lieu = df_c["lieu"].mode().iloc[0] if len(df_c) > 0 else "N/A"

        cluster_stats.append({
            "cluster":      c,
            "nb_offres":    len(df_c),
            "top_titres":   " | ".join(top_titles[:2]),
            "top_competences": ", ".join(top_comps[:4]),
            "contrat_dominant": top_contrat,
            "lieu_dominant": top_lieu,
        })

    return pd.DataFrame(cluster_stats)


# ------------------------------------------------------------------
# Visualisations
# ------------------------------------------------------------------

def plot_elbow(k_range, inertias, silhouettes, save_path: str = None):
    """Graphique méthode du coude + silhouette."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(k_range, inertias, 'o-', color='steelblue', linewidth=2)
    axes[0].set_title("Méthode du coude (Inertia)", fontsize=13)
    axes[0].set_xlabel("Nombre de clusters K")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_range, silhouettes, 's-', color='teal', linewidth=2)
    best_k = k_range[np.argmax(silhouettes)]
    axes[1].axvline(best_k, color='red', linestyle='--', label=f'Meilleur K={best_k}')
    axes[1].set_title("Score de silhouette", fontsize=13)
    axes[1].set_xlabel("Nombre de clusters K")
    axes[1].set_ylabel("Silhouette score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_clusters_2d(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    df_offres: pd.DataFrame,
    cv_embedding_2d: np.ndarray = None,
    save_path: str = None,
):
    """Visualisation 2D des clusters avec position du CV."""
    n_clusters = len(set(labels))
    colors = cm.tab10(np.linspace(0, 1, n_clusters))

    fig, ax = plt.subplots(figsize=(12, 8))

    for c in range(n_clusters):
        mask = labels == c
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[c]],
            label=f"Cluster {c} ({mask.sum()} offres)",
            alpha=0.6,
            s=30,
            edgecolors="none",
        )

    # Position du CV
    if cv_embedding_2d is not None:
        ax.scatter(
            cv_embedding_2d[0], cv_embedding_2d[1],
            c="red", s=300, marker="*",
            label="Mon CV", zorder=5, edgecolors="black", linewidth=1
        )

    ax.set_title("Clustering des offres d'emploi (PCA 2D)", fontsize=14)
    ax.set_xlabel("Composante 1")
    ax.set_ylabel("Composante 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_cluster_sizes(cluster_stats: pd.DataFrame, save_path: str = None):
    """Taille de chaque cluster."""
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        [f"C{c}" for c in cluster_stats["cluster"]],
        cluster_stats["nb_offres"],
        color="steelblue", edgecolor="white"
    )
    for bar, nb in zip(bars, cluster_stats["nb_offres"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(nb), ha="center", fontsize=10)
    ax.set_title("Nombre d'offres par cluster", fontsize=13)
    ax.set_ylabel("Nombre d'offres")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
