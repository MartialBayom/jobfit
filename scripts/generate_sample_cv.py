from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

def generate_sample_cv(output_path="data/cv/sample_cv.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(auto=True, margin=20)

    def h(txt, size=12):
        pdf.set_font("Helvetica", "B", size)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 8, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def p(txt, size=10, color=(60,60,60)):
        pdf.set_font("Helvetica", "", size)
        pdf.set_text_color(*color)
        pdf.multi_cell(0, 6, txt)

    def row(left, right, lw=50):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(40,40,40)
        pdf.cell(lw, 6, left, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(60,60,60)
        pdf.cell(0, 6, right, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # En-tête
    pdf.set_font("Helvetica", "B", 22); pdf.set_text_color(30,30,30)
    pdf.cell(0, 12, "Sophie Martin", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "B", 12); pdf.set_text_color(70,130,180)
    pdf.cell(0, 8, "DATA SCIENTIST - MACHINE LEARNING ENGINEER", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    p("sophie.martin@email.com | Paris (75) | github.com/sophiemartin", color=(80,80,80))
    pdf.ln(3)
    pdf.set_draw_color(200,200,200); pdf.line(20, pdf.get_y(), 190, pdf.get_y()); pdf.ln(4)

    # Profil
    h("Profil")
    p("Data Scientist avec 3 ans d experience en machine learning et NLP. "
      "Master Statistiques et Data Science Paris Saclay. "
      "Specialisee dans le traitement du langage naturel, la modelisation predictive "
      "et le deploiement de modeles en production via FastAPI et Docker. "
      "Disponible CDI Paris ou remote.")
    pdf.ln(3)

    # Compétences
    h("Competences")
    for cat, skills in [
        ("Langages",         "Python, SQL, R, Scala"),
        ("Machine Learning", "Scikit-learn, XGBoost, LightGBM, Random Forest, GridSearchCV"),
        ("NLP",              "spaCy, NLTK, Transformers, BERT, TF-IDF, Hugging Face, Embeddings"),
        ("Deep Learning",    "TensorFlow, Keras, PyTorch"),
        ("Big Data",         "PySpark, Databricks, Kafka, Airflow"),
        ("Cloud & MLOps",    "AWS, AWS S3, Docker, MLflow, FastAPI, Streamlit"),
        ("Analyse & Viz",    "Pandas, NumPy, Matplotlib, Seaborn, Plotly, Power BI"),
        ("Bases de donnees", "PostgreSQL, MongoDB, SQLAlchemy"),
        ("Outils",           "Jupyter, GitHub, Git, API REST"),
    ]:
        row(cat + " :", skills)
    pdf.ln(3)

    # Expériences
    h("Experiences")
    for titre, periode, desc, tech in [
        ("Data Scientist - BNP Paribas", "2022 - Aujourd hui",
         "Modeles scoring credit XGBoost LightGBM. Pipeline NLP documents clients BERT spaCy. "
         "Deploiement FastAPI Docker AWS. Suivi MLflow.",
         "Python, XGBoost, BERT, FastAPI, Docker, AWS, MLflow, SQL"),
        ("Data Analyst - Decathlon", "2021 - 2022",
         "Analyse ventes et comportements clients. Dashboards Power BI. "
         "Clustering KMeans DBSCAN. SQL PostgreSQL.",
         "Python, SQL, PostgreSQL, Power BI, Scikit-learn, Pandas"),
    ]:
        pdf.set_font("Helvetica", "B", 11); pdf.set_text_color(40,40,40)
        pdf.cell(130, 7, titre, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Helvetica", "I", 10); pdf.set_text_color(100,100,100)
        pdf.cell(0, 7, periode, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
        p(desc); 
        pdf.set_font("Helvetica", "I", 9); pdf.set_text_color(70,130,180)
        pdf.cell(0, 6, tech, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    # Formation
    h("Formation")
    for titre, ecole, periode in [
        ("Master 2 Statistiques et Data Science", "Universite Paris-Saclay", "2019-2021"),
        ("Licence Mathematiques et Informatique",  "Universite Paris 6",     "2016-2019"),
    ]:
        row(titre, periode, lw=130)
        p("  " + ecole, color=(80,80,80))
        pdf.ln(1)

    # Langues
    pdf.ln(2); h("Langues")
    p("Francais natif | Anglais C1 | Espagnol intermediaire")

    pdf.output(output_path)
    print(f"CV fictif genere : {output_path}")

if __name__ == "__main__":
    generate_sample_cv()
