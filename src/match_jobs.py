import os
import pandas as pd
from pypdf import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

def extract_resume_text(path="resume.pdf"):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    resume_text = extract_resume_text("resume.pdf")
    resume_vector = embeddings.embed_query(resume_text)

    df = pd.read_csv("data/jobs.csv")

    scores = []
    for _, row in df.iterrows():
        jd_text = str(row["jd_text"])
        jd_vector = embeddings.embed_query(jd_text)
        score = cosine_similarity(resume_vector, jd_vector)
        scores.append(score)

    df["match_score"] = scores
    df = df.sort_values("match_score", ascending=False)
    df.to_csv("data/jobs_ranked.csv", index=False)
    print(f"Ranked {len(df)} jobs. Top match: {df.iloc[0]['title']} at {df.iloc[0]['company']} ({df.iloc[0]['match_score']:.3f})")

if __name__ == "__main__":
    main()
