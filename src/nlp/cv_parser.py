"""
src/nlp/cv_parser.py
Extraction structurée d'un CV PDF — tous secteurs.
"""

import re
import fitz
from pathlib import Path


# ── Référentiel multi-secteurs ─────────────────────────────────────
SKILLS_DICT = {
    # Tech & Data
    "langages": ["python", "sql", "r", "scala", "java", "javascript", "typescript",
                 "bash", "julia", "c++", "c#", "php", "ruby", "swift", "kotlin"],
    "data_analyse": ["pandas", "numpy", "scipy", "statsmodels", "excel", "power bi",
                     "powerbi", "tableau", "looker", "metabase", "plotly", "matplotlib",
                     "seaborn", "qlik", "sas", "spss"],
    "machine_learning": ["scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost",
                         "random forest", "gradient boosting", "svm", "régression",
                         "classification", "clustering", "kmeans", "pca", "cross-validation",
                         "deep learning", "tensorflow", "keras", "pytorch", "nlp",
                         "transformers", "bert", "llm", "rag", "langchain"],
    "cloud_devops": ["aws", "gcp", "azure", "docker", "kubernetes", "mlflow", "fastapi",
                     "streamlit", "git", "github", "gitlab", "ci/cd", "linux", "nginx"],
    "databases": ["postgresql", "mysql", "sqlite", "mongodb", "redis", "elasticsearch",
                  "oracle", "sql server", "bigquery", "snowflake"],

    # Santé & Social
    "soins": ["soins infirmiers", "soins intensifs", "bloc opératoire", "urgences",
              "pédiatrie", "gériatrie", "oncologie", "cardiologie", "neurologie",
              "psychiatrie", "réanimation", "chirurgie", "anesthésie",
              "soins palliatifs", "plaies et cicatrisation", "perfusion",
              "injection", "pansement", "cathéter", "sonde", "électrocardiogramme",
              "glycémie", "tension artérielle", "dossier patient", "ifsi",
              "aide-soignant", "infirmier", "infirmière", "sage-femme",
              "kinésithérapeute", "ergothérapeute", "psychologue", "médecin",
              "pharmacien", "biologiste"],
    "logiciels_sante": ["dpi", "mediboard", "crossway", "dx care", "hopital manager",
                        "logiciel médical", "prescription médicale"],
    "social": ["accompagnement", "éducateur spécialisé", "aide sociale", "bpjeps",
               "dejeps", "dees", "deme", "travail social", "protection enfance",
               "handicap", "insertion professionnelle"],

    # Marketing & Communication
    "marketing": ["marketing digital", "seo", "sea", "sem", "google ads", "facebook ads",
                  "emailing", "crm", "salesforce", "hubspot", "inbound marketing",
                  "content marketing", "community management", "réseaux sociaux",
                  "instagram", "linkedin", "tiktok", "branding", "identité visuelle"],
    "communication": ["relations presse", "relations publiques", "attaché de presse",
                      "rédaction", "copywriting", "journalisme", "communication interne",
                      "communication externe", "événementiel", "organisation événements"],
    "creation": ["adobe photoshop", "photoshop", "illustrator", "indesign", "after effects",
                 "premiere pro", "figma", "sketch", "canva", "motion design",
                 "graphisme", "webdesign", "ux design", "ui design", "maquettage",
                 "direction artistique", "typographie", "charte graphique"],

    # Finance & Comptabilité
    "finance": ["comptabilité", "contrôle de gestion", "audit", "finance d'entreprise",
                "analyse financière", "bilan", "compte de résultat", "trésorerie",
                "budget", "prévisionnel", "fiscalité", "consolidation", "ifrs",
                "sage", "cegid", "sap", "oracle finance", "excel financier",
                "modélisation financière", "valorisation", "m&a", "private equity",
                "banque", "assurance", "risk management", "compliance"],

    # RH & Management
    "rh": ["recrutement", "sourcing", "gestion des talents", "formation", "gpec",
           "paie", "administration du personnel", "droit du travail", "sirh",
           "workday", "successfactors", "people soft", "entretien annuel",
           "onboarding", "marque employeur", "qvt", "ats"],
    "management": ["management", "leadership", "gestion d'équipe", "coaching",
                   "conduite du changement", "gestion de projet", "agile", "scrum",
                   "kanban", "prince2", "pmp", "lean", "six sigma", "kaizen"],

    # Commercial & Vente
    "commercial": ["vente", "négociation", "prospection", "business development",
                   "account management", "key account", "grand compte", "b2b", "b2c",
                   "pipeline", "crm", "salesforce", "hubspot", "closing",
                   "chiffre d'affaires", "objectifs commerciaux"],

    # Juridique
    "juridique": ["droit des affaires", "droit social", "droit commercial",
                  "contrats", "litiges", "contentieux", "conseil juridique",
                  "propriété intellectuelle", "gdpr", "rgpd", "compliance",
                  "avocat", "juriste", "notaire"],

    # BTP & Industrie
    "btp": ["génie civil", "béton armé", "charpente", "maçonnerie", "plomberie",
            "électricité", "hvac", "autocad", "revit", "bim", "conducteur travaux",
            "chef de chantier", "métreur", "devis", "appel d'offres"],
    "industrie": ["maintenance industrielle", "productique", "qualité", "iso 9001",
                  "lean manufacturing", "amélioration continue", "automatisme",
                  "plc", "siemens", "schneider", "robotique", "usinage",
                  "soudure", "hydraulique", "pneumatique"],

    # Logistique & Transport
    "logistique": ["supply chain", "logistique", "gestion des stocks", "entrepôt",
                   "wms", "erp", "sap logistique", "transport", "douane",
                   "incoterms", "approvisionnement", "achat", "sourcing fournisseurs"],

    # Hôtellerie & Restauration
    "hotellerie": ["hôtellerie", "restauration", "cuisine", "pâtisserie", "service",
                   "sommellerie", "bar", "réception", "front office", "back office",
                   "yield management", "revenue management", "opera", "fidelio"],

    # Enseignement & Formation
    "enseignement": ["pédagogie", "enseignement", "formation professionnelle",
                     "e-learning", "ingénierie pédagogique", "lms", "moodle",
                     "tutorat", "animation", "facilitation"],

    # Langues
    "langues": ["anglais", "français", "espagnol", "allemand", "italien", "portugais",
                "mandarin", "arabe", "japonais", "toeic", "toefl", "ielts",
                "bilingue", "trilingue", "c1", "c2", "b2"],
}

ALL_SKILLS = []
for category, skills in SKILLS_DICT.items():
    for skill in skills:
        ALL_SKILLS.append((skill, category))


# ── Titres professionnels tous secteurs ───────────────────────────
TITLE_PATTERNS = [
    # Santé
    r'infirmier[e]?\b[^\n]*', r'aide.soignant[e]?\b[^\n]*',
    r'médecin\b[^\n]*', r'pharmacien[ne]?\b[^\n]*',
    r'kinésithérapeute\b[^\n]*', r'sage.femme\b[^\n]*',
    r'chirurgien[ne]?\b[^\n]*', r'anesthésiste\b[^\n]*',
    r'urgentiste\b[^\n]*', r'radiologue\b[^\n]*',
    r'psychologue\b[^\n]*', r'ergothérapeute\b[^\n]*',
    # Data & Tech
    r'data\s+scientist[^\n]*', r'data\s+analyst[^\n]*',
    r'data\s+engineer[^\n]*', r'machine\s+learning[^\n]*',
    r'développeur[^\n]*', r'developer[^\n]*',
    r'ingénieur[^\n]*logiciel[^\n]*', r'devops[^\n]*',
    r'architecte[^\n]*cloud[^\n]*', r'full.stack[^\n]*',
    # Marketing
    r'chef\s+de\s+projet\s+digital[^\n]*',
    r'community\s+manager[^\n]*',
    r'responsable\s+marketing[^\n]*',
    r'directeur[^\n]*marketing[^\n]*',
    r'traffic\s+manager[^\n]*',
    r'chargé[e]?\s+de\s+communication[^\n]*',
    r'graphiste[^\n]*', r'motion\s+designer[^\n]*',
    r'directeur[^\n]*artistique[^\n]*',
    r'ux\s+designer[^\n]*', r'ui\s+designer[^\n]*',
    # Finance
    r'contrôleur[^\n]*gestion[^\n]*',
    r'analyste\s+financier[^\n]*',
    r'comptable[^\n]*', r'auditeur[^\n]*',
    r'directeur\s+financier[^\n]*', r'daf\b[^\n]*',
    r'gestionnaire[^\n]*paie[^\n]*',
    r'conseiller[^\n]*bancaire[^\n]*',
    # RH
    r'responsable\s+rh[^\n]*', r'drh\b[^\n]*',
    r'chargé[e]?\s+de\s+recrutement[^\n]*',
    r'talent\s+acquisition[^\n]*',
    r'chargé[e]?\s+de\s+formation[^\n]*',
    # Commercial
    r'commercial[e]?\b[^\n]*', r'business\s+developer[^\n]*',
    r'account\s+manager[^\n]*', r'directeur\s+commercial[^\n]*',
    r'chargé[e]?\s+d.affaires[^\n]*',
    # Juridique
    r'juriste[^\n]*', r'avocat[e]?\b[^\n]*',
    r'notaire[^\n]*', r'paralegal[^\n]*',
    # BTP / Industrie
    r'ingénieur[^\n]*génie[^\n]*',
    r'chef\s+de\s+chantier[^\n]*',
    r'conducteur[^\n]*travaux[^\n]*',
    r'technicien[^\n]*maintenance[^\n]*',
    r'ingénieur[^\n]*production[^\n]*',
    # Logistique
    r'responsable\s+logistique[^\n]*',
    r'supply\s+chain[^\n]*',
    r'responsable\s+achat[^\n]*',
    # Enseignement
    r'professeur[^\n]*', r'enseignant[e]?\b[^\n]*',
    r'formateur[^\n]*',
    # Management
    r'directeur\s+général[^\n]*', r'dg\b[^\n]*',
    r'chef\s+de\s+projet[^\n]*',
    r'responsable[^\n]*',
    r'manager[^\n]*',
]


# ── Extraction texte PDF ───────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n"
    doc.close()
    return full_text


# ── Extraction compétences ─────────────────────────────────────────
def extract_skills(text: str) -> dict:
    text_lower = text.lower()
    found = {cat: [] for cat in SKILLS_DICT}
    found_flat = []

    for skill, category in ALL_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            if skill not in found[category]:
                found[category].append(skill)
            if skill not in found_flat:
                found_flat.append(skill)

    found = {k: v for k, v in found.items() if v}
    return {"par_categorie": found, "liste": found_flat}


# ── Extraction titre universel ─────────────────────────────────────
def extract_title(text: str) -> str:
    """Extrait le titre professionnel — tous secteurs."""
    for pattern in TITLE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            title = match.group(0).strip()
            # Nettoyer
            title = re.sub(r'\s+', ' ', title)
            title = title[:80]
            if len(title) > 3:
                return title

    # Fallback : première ligne non vide significative
    lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 5]
    for line in lines[:8]:
        if not re.match(r'^(adresse|email|tel|phone|né|nee|linkedin|\+33|0[67])', line.lower()):
            if len(line) < 80:
                return line

    return "Professionnel(le)"


# ── Extraction profil ──────────────────────────────────────────────
def extract_profile(text: str) -> str:
    match = re.search(
        r'(?:Profil|Résumé|Objectif|À propos|About)\s*[:\n]?\s*(.{50,600}?)(?=\n[A-Z]|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    if match:
        profile = re.sub(r'\s+', ' ', match.group(1).strip())
        return profile
    return ""


# ── Extraction expérience ──────────────────────────────────────────
def extract_experience_years(text: str) -> int:
    patterns = [
        r'(\d+)\s*ans?\s+d.expérience',
        r'(\d+)\s*years?\s+of\s+experience',
        r'expérience\s+de\s+(\d+)\s*ans?',
        r'(\d+)\s*ans?\s+d.ancienneté',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0


# ── Extraction formations ──────────────────────────────────────────
def extract_education(text: str) -> list:
    education = []
    keywords = ["bac+", "master", "licence", "bachelor", "rncp", "mba",
                "ingénieur", "doctorat", "bts", "dut", "formation", "ifsi",
                "diplôme", "certificat", "dees", "deme", "dejeps", "bpjeps",
                "cap", "bep", "dut", "but"]
    lines = text.split('\n')
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in keywords):
            education.append(line.strip())
    return [e for e in education if len(e) > 5][:5]


# ── Parser principal ───────────────────────────────────────────────
def parse_cv(pdf_path: str) -> dict:
    print(f"Parsing du CV : {pdf_path}")
    raw_text       = extract_text_from_pdf(pdf_path)
    skills         = extract_skills(raw_text)
    title          = extract_title(raw_text)
    profile        = extract_profile(raw_text)
    exp_years      = extract_experience_years(raw_text)
    education      = extract_education(raw_text)
    enriched_text  = f"{title}. {profile} Compétences : {', '.join(skills['liste'][:30])}."

    result = {
        "raw_text": raw_text,
        "enriched_text": enriched_text,
        "titre": title,
        "profil": profile,
        "competences": skills["liste"],
        "competences_par_categorie": skills["par_categorie"],
        "experience": exp_years,
        "experience_years": exp_years,
        "formations": education,
        "formation": education,
        "nb_competences": len(skills["liste"]),
    }

    print(f"  Titre          : {result['titre']}")
    print(f"  Compétences    : {result['nb_competences']} trouvées")
    print(f"  Expérience     : {result['experience_years']} ans")
    print(f"  Formations     : {len(education)}")
    return result


if __name__ == "__main__":
    import sys
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "../../data/cv/mon_cv.pdf"
    result = parse_cv(pdf_path)
    print("\n--- COMPÉTENCES PAR CATÉGORIE ---")
    for cat, skills in result["competences_par_categorie"].items():
        print(f"  {cat:25s}: {', '.join(skills)}")
