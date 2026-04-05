"""
src/api/france_travail.py
Gestion de l'authentification OAuth2 et récupération des offres d'emploi
via l'API France Travail.
"""

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class FranceTravailClient:
    """Client pour l'API France Travail (OAuth2 + requêtes offres)."""

    def __init__(self):
        self.client_id = os.getenv("FT_CLIENT_ID")
        self.client_secret = os.getenv("FT_CLIENT_SECRET")
        self.token_url = os.getenv("FT_TOKEN_URL")
        self.api_base = os.getenv("FT_API_BASE")
        self._token = None
        self._token_expiry = 0

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        """Obtenir (ou renouveler) le token OAuth2."""
        if self._token and time.time() < self._token_expiry - 30:
            return self._token

        response = requests.post(
            self.token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": "api_offresdemploiv2 o2dsoffre",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        data = response.json()
        self._token = data["access_token"]
        self._token_expiry = time.time() + data.get("expires_in", 1490)
        print("Token OAuth2 obtenu.")
        return self._token

    # ------------------------------------------------------------------
    # Recherche d'offres
    # ------------------------------------------------------------------

    def search_offres(
        self,
        keywords: str = "",
        code_rome: str = "",
        departement: str = "",
        max_results: int = 150,
    ) -> list[dict]:
        """
        Rechercher des offres d'emploi.

        Args:
            keywords:    Mots-clés (ex: "data scientist")
            code_rome:   Code ROME du métier (ex: "M1805")
            departement: Numéro de département (ex: "75")
            max_results: Nombre max d'offres à récupérer

        Returns:
            Liste de dicts représentant les offres
        """
        token = self._get_token()
        headers = {"Authorization": f"Bearer {token}"}

        all_offres = []
        start = 0
        page_size = 150  # max autorisé par l'API

        while len(all_offres) < max_results:
            params = {
                "range": f"{start}-{start + page_size - 1}",
            }
            if keywords:
                params["motsCles"] = keywords
            if code_rome:
                params["codeROME"] = code_rome
            if departement:
                params["departement"] = departement

            resp = requests.get(
                f"{self.api_base}/offres/search",
                headers=headers,
                params=params,
            )

            if resp.status_code == 206 or resp.status_code == 200:
                data = resp.json()
                offres = data.get("resultats", [])
                if not offres:
                    break
                all_offres.extend(offres)
                start += len(offres)
                print(f"  {len(all_offres)} offres récupérées...")
            elif resp.status_code == 204:
                print("Aucune offre trouvée pour ces critères.")
                break
            else:
                print(f"Erreur {resp.status_code}: {resp.text}")
                break

        return all_offres[:max_results]

    # ------------------------------------------------------------------
    # Conversion DataFrame
    # ------------------------------------------------------------------

    def offres_to_dataframe(self, offres: list[dict]) -> pd.DataFrame:
        """Convertir une liste d'offres en DataFrame structuré."""
        rows = []
        for o in offres:
            rows.append({
                "id": o.get("id"),
                "intitule": o.get("intitule"),
                "description": o.get("description", ""),
                "date_creation": o.get("dateCreation"),
                "entreprise": o.get("entreprise", {}).get("nom", ""),
                "lieu": o.get("lieuTravail", {}).get("libelle", ""),
                "code_postal": o.get("lieuTravail", {}).get("codePostal", ""),
                "type_contrat": o.get("typeContrat"),
                "experience": o.get("experienceLibelle", ""),
                "formation": o.get("formations", [{}])[0].get("niveauLibelle", "") if o.get("formations") else "",
                "salaire": o.get("salaire", {}).get("libelle", ""),
                "competences": ", ".join([c.get("libelle", "") for c in o.get("competences", [])]),
                "secteur_activite": o.get("secteurActiviteLibelle", ""),
                "code_rome": o.get("romeCode", ""),
            })
        return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Usage direct
# ------------------------------------------------------------------

if __name__ == "__main__":
    client = FranceTravailClient()

    print("Recherche d'offres Data Scientist en France...")
    offres = client.search_offres(
        keywords="data scientist",
        max_results=150,
    )

    df = client.offres_to_dataframe(offres)
    print(f"\n{len(df)} offres récupérées.")
    print(df[["intitule", "entreprise", "lieu", "type_contrat"]].head(10))

    # Sauvegarde
    output_path = os.getenv("DATA_RAW", "./data/raw")
    os.makedirs(output_path, exist_ok=True)
    df.to_csv(f"{output_path}/offres_raw.csv", index=False)
    print(f"\nSauvegardé dans {output_path}/offres_raw.csv")
