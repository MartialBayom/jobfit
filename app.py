"""app.py - JobFit Dashboard"""
import os, sys, sqlite3, warnings
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data/processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CV_DIR     = os.path.join(BASE_DIR, "data/cv")

sys.path.insert(0, os.path.join(BASE_DIR, "src/nlp"))
sys.path.insert(0, os.path.join(BASE_DIR, "src/ml"))
sys.path.insert(0, os.path.join(BASE_DIR, "src/rag"))
sys.path.insert(0, os.path.join(BASE_DIR, "src/api"))

st.set_page_config(page_title="JobFit", page_icon="logo.png", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0a0a0f; }
[data-testid="stSidebar"] { background: #0f0f1a !important; border-right: 1px solid #1e1e3a; }
[data-testid="stSidebar"] * { color: #a0a8c8 !important; }
.block-container { padding: 2rem 2.5rem; }
h1,h2,h3,h4 { color: #e8eaf6 !important; font-weight: 700 !important; }
p,li { color: #6b7299; }
.jf-sub { color: #4a5180; font-size: 1rem; margin-bottom: 2rem; }
.jf-card { background: #12121f; border: 1px solid #1e1e3a; border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem; }
.jf-step { display: flex; align-items: flex-start; gap: 1rem; background: #12121f; border: 1px solid #1e1e3a; border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 0.8rem; }
.jf-step-num { font-size: 1.5rem; font-weight: 800; background: linear-gradient(135deg, #7c4dff, #00d4aa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; min-width: 32px; }
.skill-tag { background: #00d4aa15; color: #00d4aa; border: 1px solid #00d4aa30; padding: 3px 10px; border-radius: 20px; font-size: 0.78rem; margin: 2px; display: inline-block; font-weight: 500; }
.missing-tag { background: #e74c3c15; color: #e74c3c; border: 1px solid #e74c3c30; padding: 3px 10px; border-radius: 20px; font-size: 0.78rem; margin: 2px; display: inline-block; font-weight: 500; }
.score-box { background: #12121f; border: 1px solid #1e1e3a; border-left: 4px solid #7c4dff; padding: 1.2rem 1.5rem; border-radius: 12px; margin: 0.5rem 0; }
.score-box h2, .score-box h3 { color: #e8eaf6 !important; margin: 0 0 0.4rem; }
.score-box p { color: #6b7299; margin: 0.15rem 0; font-size: 0.9rem; }
.score-high { border-left-color: #00d4aa !important; }
.score-mid  { border-left-color: #f39c12 !important; }
.score-low  { border-left-color: #e74c3c !important; }
[data-testid="stTextInput"] input { background: #12121f !important; border: 1px solid #1e1e3a !important; color: #e8eaf6 !important; border-radius: 10px !important; }
.stButton > button { background: #1e1e3a !important; color: #e8eaf6 !important; border: 1px solid #7c4dff !important; border-radius: 10px !important; font-weight: 700 !important; padding: 0.6rem 1.8rem !important; }
.stButton > button:hover { background: #7c4dff !important; color: #ffffff !important; }
[data-testid="metric-container"] { background: #12121f; border: 1px solid #1e1e3a; border-radius: 14px; padding: 1rem 1.2rem; }
[data-testid="stMetricValue"] { color: #e8eaf6 !important; font-size: 1.8rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #4a5180 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricDelta"] { color: #00d4aa !important; }
hr { border-color: #1e1e3a !important; }
[data-testid="stAlert"] { background: #12121f !important; border: 1px solid #1e1e3a !important; border-radius: 10px !important; }
[data-testid="stDataFrame"] { border: 1px solid #1e1e3a; border-radius: 12px; overflow: hidden; }
a { color: #00d4aa !important; }
</style>
""", unsafe_allow_html=True)


# ── Cache ──────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    from embeddings import EmbeddingEngine
    engine = EmbeddingEngine()
    scorer = joblib.load(os.path.join(MODELS_DIR, "best_scorer.pkl"))
    km     = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    return engine, scorer, km

@st.cache_data
def load_data():
    conn  = sqlite3.connect(os.path.join(DATA_DIR, "jobfit.db"))
    df    = pd.read_sql("SELECT * FROM offres", conn)
    conn.close()
    embs  = np.load(os.path.join(DATA_DIR, "offres_embeddings.npy"))
    embs2 = np.load(os.path.join(DATA_DIR, "embeddings_2d.npy"))
    return df, embs, embs2


# ── Helpers ────────────────────────────────────────────────────────
def score_color(s):
    return "score-high" if s >= 70 else ("score-mid" if s >= 50 else "score-low")

def score_emoji(s):
    return "🟢" if s >= 70 else ("🟡" if s >= 50 else "🔴")

def format_contrat(c):
    return {"MIS":"Intérim","LIB":"Libéral","SAI":"Saisonnier",
            "CCE":"Alternance","DDI":"Insertion","FRA":"Franchise","DIN":"Intérim"}.get(c, c)

def get_missing(cv_data, offre_row):
    import re
    from cv_parser import ALL_SKILLS
    txt    = f"{offre_row.get('intitule','')} {offre_row.get('description','')} {offre_row.get('competences','')}"
    cv_set = set(s.lower() for s in cv_data["competences"])
    return [sk for sk,_ in ALL_SKILLS
            if sk not in cv_set and re.search(r'\b'+re.escape(sk)+r'\b', txt.lower())][:8]

def parse_cv_from_upload(uploaded_file):
    import tempfile
    from cv_parser import parse_cv
    raw = uploaded_file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(raw); tmp_path = tmp.name
    cv = parse_cv(tmp_path)
    os.unlink(tmp_path)
    return cv

def format_top_offres(df):
    out = df.copy()
    if "date_creation" in out.columns:
        out["Date"] = pd.to_datetime(out["date_creation"], errors="coerce").dt.strftime("%d/%m/%Y")
    else:
        out["Date"] = "—"
    out["Contrat"] = out["type_contrat"].apply(format_contrat)
    out["Score"]   = out["ml_score_pct"].apply(lambda x: f"{score_emoji(x)} {x}%")
    return out


# ── Génération IA ──────────────────────────────────────────────────
def generate_with_claude(prompt, max_tokens=2000):
    try:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None, "Cle ANTHROPIC_API_KEY manquante"
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text, None
    except Exception as e:
        return None, str(e)

def generate_cv_ats(cv_data, offre_dict):
    competences = ", ".join(cv_data.get("competences", []))
    offre_desc  = f"{offre_dict.get('intitule','')} — {offre_dict.get('description','')[:500]}"
    prompt = f"""Tu es un expert ATS. Genere un CV optimise ATS en francais.

CV candidat :
- Titre : {cv_data.get('titre','')}
- Competences : {competences}
- Experience : {cv_data.get('experience',0)} ans

Offre cible : {offre_desc}

Structure (titres en MAJUSCULES) :
PROFIL PROFESSIONNEL
COMPETENCES TECHNIQUES
EXPERIENCES PROFESSIONNELLES
FORMATIONS
COMPETENCES TRANSVERSALES

Regles : mots-cles exacts de l'offre, format texte pur, bullet points avec tiret.
Reponds UNIQUEMENT avec le CV."""
    return generate_with_claude(prompt, 1500)

def generate_cover_letter(cv_data, offre_dict):
    competences = ", ".join(cv_data.get("competences", [])[:10])
    prompt = f"""Tu es expert en lettres de motivation. Redige une lettre en francais.

Candidat :
- Titre : {cv_data.get('titre','')}
- Competences : {competences}
- Experience : {cv_data.get('experience',0)} ans

Offre :
- Poste : {offre_dict.get('intitule','')}
- Entreprise : {offre_dict.get('entreprise','')}
- Description : {offre_dict.get('description','')[:400]}

Structure : en-tete, objet, introduction, corps (2 §), conclusion, politesse.
250-300 mots max. Reponds UNIQUEMENT avec la lettre."""
    return generate_with_claude(prompt, 1000)

def create_pdf(title, content_text):
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_margins(20, 20, 20)
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(30, 30, 80)
        pdf.cell(0, 10, title[:80], ln=True)
        pdf.ln(2)
        pdf.set_draw_color(124, 77, 255)
        pdf.set_line_width(0.5)
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(5)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        for line in content_text.split("\n"):
            line = line.strip()
            if not line:
                pdf.ln(3); continue
            if line.isupper() and len(line) > 3:
                pdf.ln(2)
                pdf.set_font("Helvetica", "B", 11)
                pdf.set_text_color(100, 70, 200)
                pdf.cell(0, 7, line, ln=True)
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(40, 40, 40)
            elif line.startswith("-") or line.startswith("•"):
                pdf.set_x(25)
                pdf.multi_cell(0, 5, line)
            else:
                pdf.multi_cell(0, 5, line)
        return bytes(pdf.output()), None
    except Exception as e:
        return None, str(e)

def show_generation_section(cv_data, offre_dict, prefix="main"):
    """Section génération avec session_state pour éviter le rechargement."""
    st.divider()
    st.markdown("#### 🤖 Génération automatique")
    c1, c2 = st.columns(2)

    key_cv  = f"{prefix}_cv_text"
    key_lm  = f"{prefix}_lm_text"

    with c1:
        if st.button("📄 Générer CV optimisé ATS", key=f"btn_cv_{prefix}"):
            with st.spinner("Génération en cours..."):
                text, err = generate_cv_ats(cv_data, offre_dict)
            if err:
                st.error(f"Erreur : {err}")
            else:
                st.session_state[key_cv] = text

        if st.session_state.get(key_cv):
            st.success("CV ATS généré !")
            st.text_area("CV optimisé ATS", st.session_state[key_cv], height=350, key=f"ta_cv_{prefix}")
            pdf, _ = create_pdf(f"CV ATS — {offre_dict.get('intitule','')}", st.session_state[key_cv])
            if pdf:
                st.download_button("⬇️ Télécharger CV PDF", data=pdf,
                    file_name="cv_ats_jobfit.pdf", mime="application/pdf", key=f"dl_cv_{prefix}")

    with c2:
        if st.button("✉️ Générer lettre de motivation", key=f"btn_lm_{prefix}"):
            with st.spinner("Génération en cours..."):
                text, err = generate_cover_letter(cv_data, offre_dict)
            if err:
                st.error(f"Erreur : {err}")
            else:
                st.session_state[key_lm] = text

        if st.session_state.get(key_lm):
            st.success("Lettre générée !")
            st.text_area("Lettre de motivation", st.session_state[key_lm], height=350, key=f"ta_lm_{prefix}")
            pdf, _ = create_pdf("Lettre de motivation — JobFit", st.session_state[key_lm])
            if pdf:
                st.download_button("⬇️ Télécharger lettre PDF", data=pdf,
                    file_name="lettre_motivation_jobfit.pdf", mime="application/pdf", key=f"dl_lm_{prefix}")


# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.image("logo.png", width=180)
    st.divider()
    page = st.radio("Nav", [
        "🏠 Accueil",
        "📄 Analyser mon CV",
        "✏️ Matcher avec une offre",
        "📊 Explorer le marché"
    ], label_visibility="collapsed")
    st.divider()
    st.caption("Jedha AI School · 2026")
    st.caption("NLP · ML · RAG · MLOps")


# ── Accueil ────────────────────────────────────────────────────────
if page == "🏠 Accueil":
    st.image("logo.png", width=300)
    st.markdown('<p class="jf-sub">Trouve les offres qui correspondent à ton profil · Génération automatique CV & Lettre</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Offres indexées", "7 002", "France Travail API")
    with c2: st.metric("Secteurs couverts", "15+", "Tous domaines")
    with c3: st.metric("Analyse", "Temps réel", "Matching intelligent")
    with c4: st.metric("Génération", "CV + Lettre", "Optimisés ATS")

    st.divider()
    st.markdown("#### Comment ça marche ?")
    st.markdown("""
    <div class="jf-step"><div class="jf-step-num">1</div><div>
        <div style="color:#e8eaf6;font-weight:700;margin-bottom:0.3rem">Upload ton CV</div>
        <div style="color:#4a5180;font-size:0.87rem">Charge ton PDF — JobFit extrait automatiquement tes compétences, ton titre et ton expérience.</div>
    </div></div>
    <div class="jf-step"><div class="jf-step-num">2</div><div>
        <div style="color:#e8eaf6;font-weight:700;margin-bottom:0.3rem">Trouve les offres qui matchent</div>
        <div style="color:#4a5180;font-size:0.87rem">JobFit interroge l'API France Travail en temps réel et score les offres selon ton profil.</div>
    </div></div>
    <div class="jf-step"><div class="jf-step-num">3</div><div>
        <div style="color:#e8eaf6;font-weight:700;margin-bottom:0.3rem">Colle une offre spécifique</div>
        <div style="color:#4a5180;font-size:0.87rem">Tu as trouvé une offre sur LinkedIn ? Colle-la pour voir ton score et obtenir des conseils.</div>
    </div></div>
    <div class="jf-step"><div class="jf-step-num">4</div><div>
        <div style="color:#e8eaf6;font-weight:700;margin-bottom:0.3rem">Génère ton CV ATS + Lettre</div>
        <div style="color:#4a5180;font-size:0.87rem">Génération automatique d'un CV optimisé ATS et d'une lettre de motivation personnalisée.</div>
    </div></div>
    """, unsafe_allow_html=True)


# ── Analyser mon CV ────────────────────────────────────────────────
elif page == "📄 Analyser mon CV":
    st.title("📄 Analyser mon CV")

    uploaded_file = st.file_uploader("Charge ton CV (PDF)", type=["pdf"])
    use_sample    = st.checkbox("Utiliser le CV fictif (sample_cv.pdf)", value=False)

    if use_sample and not uploaded_file:
        p = os.path.join(CV_DIR, "sample_cv.pdf")
        if os.path.exists(p):
            with open(p, "rb") as f:
                data = f.read()
            uploaded_file = type('obj', (object,), {'read': lambda self=None: data, 'name': 'sample_cv.pdf'})()

    st.divider()
    mode = st.radio("Mode de recherche", [
        "🗄️ Base locale (7002 offres — rapide)",
        "🌐 Temps réel (API France Travail — tous secteurs)"
    ], help="Le mode temps réel interroge l'API en direct.")
    mode_realtime = "Temps réel" in mode

    if mode_realtime:
        max_rt = st.slider("Nombre d'offres à récupérer", 50, 200, 100)
        kw_rt  = st.text_input("Mots-clés", value="",
                                placeholder="infirmier, data scientist, graphiste...")
        st.info("⏱️ Le mode temps réel prend ~20 secondes.")

    if not uploaded_file:
        st.info("👆 Charge ton CV pour commencer l'analyse.")
        st.stop()

    # Bouton analyser
    if st.button("🔍 Lancer l'analyse", type="primary") or "analyse_cv_done" in st.session_state:
        if "analyse_cv_done" not in st.session_state:
            with st.spinner("Analyse en cours..."):
                try:
                    engine, scorer, km_model = load_models()

                    if mode_realtime:
                        if not kw_rt:
                            st.warning("Entre des mots-clés pour la recherche.")
                            st.stop()
                        from france_travail import FranceTravailClient
                        cv_data      = parse_cv_from_upload(uploaded_file)
                        cv_embedding = engine.embed_text(cv_data["enriched_text"])
                        client       = FranceTravailClient()
                        offres       = client.search_offres(keywords=kw_rt, max_results=max_rt)
                        df_rt        = client.offres_to_dataframe(offres)
                        if df_rt.empty:
                            st.warning("Aucune offre trouvée.")
                            st.stop()
                        st.success(f"✅ {len(df_rt)} offres récupérées !")
                        with st.spinner("Calcul des scores..."):
                            textes  = [engine.prepare_offre_text(r) for _, r in df_rt.iterrows()]
                            embs_rt = engine.embed_texts(textes)
                        df_rt["ml_score_pct"] = [round(float(np.dot(cv_embedding, e)) * 100, 1) for e in embs_rt]
                        df_scored = df_rt.sort_values("ml_score_pct", ascending=False)
                    else:
                        df_offres, offres_embs, _ = load_data()
                        cv_data      = parse_cv_from_upload(uploaded_file)
                        cv_embedding = engine.embed_text(cv_data["enriched_text"])
                        # Scoring universel par similarité cosinus
                        scores = [round(float(np.dot(cv_embedding, emb)) * 100, 1) for emb in offres_embs]
                        df_scored = df_offres.copy()
                        df_scored["ml_score_pct"] = scores
                        df_scored = df_scored.sort_values("ml_score_pct", ascending=False)

                    st.session_state["analyse_cv_done"]    = True
                    st.session_state["cv_data"]            = cv_data
                    st.session_state["cv_embedding"]       = cv_embedding
                    st.session_state["df_scored"]          = df_scored
                    st.session_state["km_model"]           = km_model

                except Exception as e:
                    st.error(f"Erreur : {e}")
                    st.stop()

        # Afficher résultats depuis session_state
        cv_data      = st.session_state["cv_data"]
        cv_embedding = st.session_state["cv_embedding"]
        df_scored    = st.session_state["df_scored"]
        km_model     = st.session_state["km_model"]

        st.success("✅ CV analysé !")

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Titre", cv_data["titre"][:28])
        with c2: st.metric("Compétences", cv_data["nb_competences"])
        with c3:
            try:
                cl = km_model.predict(cv_embedding.reshape(1,-1))[0]
                st.metric("Cluster", f"Cluster {cl}")
            except: st.metric("Cluster", "N/A")

        st.markdown("#### Compétences extraites")
        for cat, skills in cv_data["competences_par_categorie"].items():
            if skills:
                tags = " ".join([f'<span class="skill-tag">{s}</span>' for s in skills])
                st.markdown(f"**{cat}** : {tags}", unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 🏆 Top 15 offres compatibles")
        top15 = format_top_offres(df_scored.head(15))
        has_url = "url" in top15.columns and top15["url"].notna().any()
        if has_url:
            top15["Postuler"] = top15["url"].apply(
                lambda u: u if u and pd.notna(u) and str(u).startswith("http") else None)
            top15_show = top15[["intitule","entreprise","lieu","Contrat","Date","Score","Postuler"]].copy()
            top15_show.columns = ["Intitulé","Entreprise","Lieu","Contrat","Date","Score","Postuler"]
            st.dataframe(top15_show,
                column_config={"Postuler": st.column_config.LinkColumn("Postuler ↗")},
                use_container_width=True, hide_index=True)
        else:
            top15_show = top15[["intitule","entreprise","lieu","Contrat","Date","Score"]].copy()
            top15_show.columns = ["Intitulé","Entreprise","Lieu","Contrat","Date","Score"]
            st.dataframe(top15_show, use_container_width=True, hide_index=True)

        st.markdown("#### 🔍 Détail d'une offre")
        idx = st.selectbox("Sélectionne une offre", range(min(20, len(df_scored))),
            format_func=lambda i: f"{df_scored.iloc[i]['intitule']} — {df_scored.iloc[i]['ml_score_pct']}%",
            key="sel_offre_cv")

        offre_sel = df_scored.iloc[idx]
        miss      = get_missing(cv_data, offre_sel)
        score     = offre_sel["ml_score_pct"]
        url       = offre_sel.get("url","")
        lien      = f'<br><a href="{url}" target="_blank">🔗 Postuler</a>' if url and pd.notna(url) and str(url).startswith("http") else ""
        date_pub  = pd.to_datetime(offre_sel.get("date_creation",""), errors="coerce")
        date_str  = date_pub.strftime("%d/%m/%Y") if pd.notna(date_pub) else "—"

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="score-box {score_color(score)}">
                <h3>{score_emoji(score)} Score : {score}%</h3>
                <p><b>{offre_sel['intitule']}</b></p>
                <p>{offre_sel.get('entreprise','')} | {offre_sel.get('lieu','')} | {format_contrat(offre_sel.get('type_contrat',''))}</p>
                <p style="color:#4a5180;font-size:0.8rem">Publié le : {date_str}</p>
                {lien}
            </div>""", unsafe_allow_html=True)
        with c2:
            if miss:
                st.markdown("**Compétences manquantes :**")
                st.markdown(" ".join([f'<span class="missing-tag">{s}</span>' for s in miss]), unsafe_allow_html=True)
            else:
                st.success("Ton profil couvre toutes les compétences !")

        st.markdown("#### 💬 Conseils de candidature")
        from chatbot import JobFitChatbot
        advice_key = f"advice_{idx}"
        if advice_key not in st.session_state:
            st.session_state[advice_key] = JobFitChatbot(cv_data).analyze_offre(offre_sel.to_dict(), score)
        st.text_area("Analyse", st.session_state[advice_key], height=200, key=f"ta_advice_{idx}")

        show_generation_section(cv_data, offre_sel.to_dict(), prefix=f"cv_{idx}")

        if st.button("🔄 Nouvelle analyse"):
            for key in ["analyse_cv_done","cv_data","cv_embedding","df_scored","km_model"]:
                st.session_state.pop(key, None)
            st.rerun()


# ── Matcher avec une offre ─────────────────────────────────────────
elif page == "✏️ Matcher avec une offre":
    st.title("✏️ Matcher avec une offre")
    st.markdown("Upload ton CV et colle une offre trouvée sur LinkedIn, Indeed ou France Travail.")

    c1, c2 = st.columns(2)
    with c1:
        uploaded_cv = st.file_uploader("Ton CV (PDF)", type=["pdf"])
        use_sample  = st.checkbox("Utiliser le CV fictif (sample_cv.pdf)", value=False)
        if use_sample and not uploaded_cv:
            p = os.path.join(CV_DIR, "sample_cv.pdf")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    data = f.read()
                uploaded_cv = type('obj', (object,), {'read': lambda self=None: data, 'name': 'sample_cv.pdf'})()
        url_offre = st.text_input("Lien de l'offre (optionnel)", placeholder="https://www.linkedin.com/jobs/...")

    with c2:
        offre_texte = st.text_area("Colle ici la description complète de l'offre", height=250,
            placeholder="Colle ici le texte complet de l'offre...")

    st.divider()

    if st.button("🎯 Analyser le match", type="primary"):
        if not uploaded_cv:
            st.warning("Upload ton CV ou coche 'Utiliser le CV fictif'.")
        elif not offre_texte.strip():
            st.warning("Colle la description de l'offre.")
        else:
            with st.spinner("Analyse en cours..."):
                engine, _, _ = load_models()
                cv_data      = parse_cv_from_upload(uploaded_cv)
                cv_embedding = engine.embed_text(cv_data["enriched_text"])
                offre_dict   = {
                    "intitule": offre_texte.split("\n")[0][:80],
                    "description": offre_texte,
                    "entreprise": "", "lieu": "",
                    "type_contrat": "", "competences": "",
                    "url": url_offre
                }
                offre_emb = engine.embed_text(offre_texte)
                score     = float(np.dot(cv_embedding, offre_emb)) * 100
                miss      = get_missing(cv_data, offre_dict)

            st.session_state["match_result"] = {
                "cv_data": cv_data, "offre_dict": offre_dict,
                "score": score, "miss": miss
            }

    if "match_result" in st.session_state:
        r        = st.session_state["match_result"]
        cv_data  = r["cv_data"]
        offre_dict = r["offre_dict"]
        score    = r["score"]
        miss     = r["miss"]
        url_offre = offre_dict.get("url","")

        lien_html = f'<p><a href="{url_offre}" target="_blank">🔗 Voir l\'offre originale</a></p>' if url_offre else ''
        st.markdown(f"""
        <div class="score-box {score_color(score)}">
            <h2>{score_emoji(score)} Score de compatibilité : {score:.1f}%</h2>
            {lien_html}
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        import re as _re
        with c1:
            st.markdown("**Compétences matchées :**")
            matched = [s for s in cv_data["competences"]
                       if _re.search(r'\b'+_re.escape(s)+r'\b', offre_dict["description"].lower())]
            if matched:
                st.markdown(" ".join([f'<span class="skill-tag">{s}</span>' for s in matched]),
                            unsafe_allow_html=True)
            else:
                st.info("Aucune compétence directement identifiée.")
        with c2:
            if miss:
                st.markdown("**Compétences manquantes :**")
                st.markdown(" ".join([f'<span class="missing-tag">{s}</span>' for s in miss]),
                            unsafe_allow_html=True)
            else:
                st.success("Ton profil couvre toutes les compétences !")

        st.markdown("#### 💬 Conseils pour optimiser ta candidature")
        if "match_advice" not in st.session_state:
            from chatbot import JobFitChatbot
            st.session_state["match_advice"] = JobFitChatbot(cv_data).analyze_offre(offre_dict, score)
        st.text_area("Analyse", st.session_state["match_advice"], height=220)

        show_generation_section(cv_data, offre_dict, prefix="match")

        if st.button("🔄 Nouvelle analyse"):
            st.session_state.pop("match_result", None)
            st.session_state.pop("match_advice", None)
            st.rerun()


# ── Explorer le marché ─────────────────────────────────────────────
elif page == "📊 Explorer le marché":
    st.title("📊 Explorer le marché de l'emploi en France")
    df_offres, _, embeddings_2d = load_data()
    _, _, km_model = load_models()

    st.markdown("#### Filtres")
    c1, c2, c3 = st.columns(3)
    with c1:
        contrat_filter = st.multiselect("Type de contrat",
            options=["CDI","CDD","Intérim","Alternance","Stage","Libéral"], default=[])
    with c2:
        ville_search = st.text_input("Ville", placeholder="Paris, Lyon, Toulouse...")
    with c3:
        keyword_filter = st.text_input("Mots-clés intitulé", placeholder="data, graphiste, infirmier...")

    contrat_map_rev = {"Intérim":"MIS","Alternance":"CCE","Stage":"SAI","Libéral":"LIB"}
    df_f = df_offres.copy()
    if contrat_filter:
        db_vals = [contrat_map_rev.get(c,c) for c in contrat_filter]
        df_f = df_f[df_f["type_contrat"].isin(db_vals)]
    if ville_search:
        df_f = df_f[df_f["lieu"].str.contains(ville_search, case=False, na=False)]
    if keyword_filter:
        df_f = df_f[df_f["intitule"].str.contains(keyword_filter, case=False, na=False)]

    st.divider()
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Offres filtrées", len(df_f))
    with c2: st.metric("CDI", len(df_f[df_f["type_contrat"]=="CDI"]))
    with c3: st.metric("CDD", len(df_f[df_f["type_contrat"]=="CDD"]))
    with c4: st.metric("Intérim", len(df_f[df_f["type_contrat"]=="MIS"]))

    st.divider()

    if not (contrat_filter or ville_search or keyword_filter):
        st.info("👆 Utilise les filtres ci-dessus pour explorer le marché.")
        st.stop()

    import plotly.express as px
    from collections import Counter

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### Répartition par contrat")
        cnt = df_f["type_contrat"].value_counts().head(8).reset_index()
        cnt.columns = ["contrat","nb"]
        cnt["contrat"] = cnt["contrat"].apply(format_contrat)
        fig = px.bar(cnt, x="contrat", y="nb", color="contrat",
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     labels={"nb":"Offres","contrat":""}, height=300)
        fig.update_layout(showlegend=False, margin=dict(t=10,b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8eaf6")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Top 10 villes")
        top_v = df_f["lieu"].value_counts().head(10).reset_index()
        top_v.columns = ["ville","nb"]
        fig = px.bar(top_v, x="nb", y="ville", orientation="h",
                     color="nb", color_continuous_scale="Teal",
                     labels={"nb":"Offres","ville":""}, height=300)
        fig.update_layout(margin=dict(t=10,b=10), coloraxis_showscale=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8eaf6")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### ☁️ Compétences les plus demandées")
    all_comps = []
    for comp_str in df_f["competences"].dropna():
        for c in comp_str.split(","):
            c = c.strip().lower()
            if c and len(c) > 2: all_comps.append(c)
    top_comps = Counter(all_comps).most_common(25)
    if top_comps:
        comp_df = pd.DataFrame(top_comps, columns=["competence","count"])
        fig = px.bar(comp_df.sort_values("count"), x="count", y="competence",
                     orientation="h", color="count", color_continuous_scale="Purples",
                     labels={"count":"Fréquence","competence":""}, height=500)
        fig.update_layout(margin=dict(t=10,b=10), coloraxis_showscale=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8eaf6")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### 📅 Offres publiées dans le temps")
    df_d = df_f.copy()
    df_d["date_creation"] = pd.to_datetime(df_d["date_creation"], errors="coerce")
    df_d = df_d.dropna(subset=["date_creation"])
    if not df_d.empty:
        by_day = df_d.groupby(df_d["date_creation"].dt.date).size().reset_index()
        by_day.columns = ["date","nb"]
        fig = px.area(by_day, x="date", y="nb", color_discrete_sequence=["#00d4aa"],
                      labels={"nb":"Offres","date":"Date"}, height=280)
        fig.update_layout(margin=dict(t=10,b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8eaf6")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### 🗺️ Carte sémantique des offres")
    labels = km_model.predict(np.load(os.path.join(DATA_DIR, "offres_embeddings.npy")))
    df_viz = df_offres.copy()
    df_viz["cluster"] = [f"Cluster {l}" for l in labels]
    df_viz["x"] = embeddings_2d[:,0]
    df_viz["y"] = embeddings_2d[:,1]
    df_viz["intitule_short"] = df_viz["intitule"].str[:40]
    fig = px.scatter(df_viz, x="x", y="y", color="cluster",
        hover_data={"intitule_short":True,"entreprise":True,"lieu":True,"type_contrat":True,"x":False,"y":False},
        opacity=0.6, height=520,
        color_discrete_sequence=px.colors.qualitative.Set1,
        labels={"x":"Dim 1","y":"Dim 2","cluster":"Cluster"})
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(margin=dict(t=10,b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#e8eaf6", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### 📋 Offres correspondantes")
    df_show = df_f.head(50).copy()
    if "date_creation" in df_show.columns:
        df_show["Date"] = pd.to_datetime(df_show["date_creation"], errors="coerce").dt.strftime("%d/%m/%Y")
    df_show["Contrat"] = df_show["type_contrat"].apply(format_contrat)
    st.dataframe(df_show[["intitule","entreprise","lieu","Contrat","experience","Date"]].rename(
        columns={"intitule":"Intitulé","entreprise":"Entreprise","lieu":"Lieu","experience":"Expérience"}),
        use_container_width=True, hide_index=True)
