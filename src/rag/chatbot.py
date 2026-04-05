"""
src/rag/chatbot.py
Conseils personnalisés basés sur Claude API.
Fallback template si pas de clé API.
"""

import os
from dotenv import load_dotenv
load_dotenv()


class JobFitChatbot:
    def __init__(self, cv_data: dict):
        self.cv_data = cv_data
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

    def analyze_offre(self, offre: dict, score: float) -> str:
        """Génère des conseils personnalisés pour une offre."""
        if self.api_key:
            return self._analyze_with_claude(offre, score)
        return self._analyze_template(offre, score)

    def _analyze_with_claude(self, offre: dict, score: float) -> str:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            competences = ", ".join(self.cv_data.get("competences", [])[:15])
            titre_cv    = self.cv_data.get("titre", "")
            exp         = self.cv_data.get("experience_years", 0)

            prompt = f"""Tu es un coach carrière expert en recrutement et optimisation ATS.

PROFIL DU CANDIDAT :
- Titre : {titre_cv}
- Compétences : {competences}
- Expérience : {exp} ans
- Score de compatibilité avec l'offre : {score:.1f}%

OFFRE D'EMPLOI :
- Poste : {offre.get('intitule', '')}
- Entreprise : {offre.get('entreprise', '')}
- Lieu : {offre.get('lieu', '')}
- Contrat : {offre.get('type_contrat', '')}
- Description : {offre.get('description', '')[:600]}
- Compétences requises : {offre.get('competences', '')}

Génère des conseils pratiques et personnalisés en 3 parties :

1. ANALYSE DU MATCH (2-3 phrases sur pourquoi ce score)
2. CONSEILS POUR OPTIMISER LE CV ATS (3 bullet points concrets basés sur les mots-clés de cette offre)
3. PROCHAINES ÉTAPES (2-3 actions concrètes pour postuler)

Sois direct, concret et personnalisé. Utilise les informations réelles de l'offre.
Réponds en français, format court (150 mots max)."""

            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}]
            )
            return msg.content[0].text

        except Exception as e:
            return self._analyze_template(offre, score) + f"\n\n[Note: {str(e)}]"

    def _analyze_template(self, offre: dict, score: float) -> str:
        """Conseils basés sur template sans API."""
        titre_offre = offre.get('intitule', 'ce poste')
        entreprise  = offre.get('entreprise', 'cette entreprise')
        competences_offre = offre.get('competences', '').split(',')[:3]
        cv_skills   = set(self.cv_data.get("competences", []))

        # Compétences manquantes spécifiques à l'offre
        manquantes = [c.strip() for c in competences_offre if c.strip().lower() not in cv_skills][:3]

        if score >= 70:
            analyse = f"Excellent match ({score:.0f}%) ! Ton profil correspond bien à {titre_offre}."
        elif score >= 50:
            analyse = f"Bon potentiel ({score:.0f}%) pour {titre_offre}. Quelques ajustements peuvent renforcer ta candidature."
        else:
            analyse = f"Match partiel ({score:.0f}%). Concentre-toi sur les compétences clés de l'offre."

        conseils = ["• Personnalise ton CV avec les mots-clés exacts de l'offre"]
        if manquantes:
            conseils.append(f"• Mets en avant : {', '.join(manquantes)}")
        conseils.append("• Quantifie tes réalisations (ex: géré X patients, réduit les délais de Y%)")

        etapes = [
            f"→ Personnalise ton CV pour faire ressortir les mots-clés de l'offre",
            f"→ Recherche {entreprise} sur LinkedIn avant l'entretien",
            f"→ Prépare 2-3 exemples concrets de tes réalisations"
        ]

        return analyse + "\n\nConseils ATS :\n" + "\n".join(conseils) + "\n\nProchaine étape :\n" + "\n".join(etapes)
