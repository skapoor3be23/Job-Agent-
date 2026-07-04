import os
import pandas as pd
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def extract_resume_text(path="resume.pdf"):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def analyze_gap(llm, resume_text, jd_text, title, company):
    prompt = f"""You are a technical recruiter. Compare this resume against this job description.

RESUME:
{resume_text}

JOB DESCRIPTION ({title} at {company}):
{jd_text}

Return:
1. Missing skills/keywords (list, max 8)
2. Specific resume edits to add (max 3 bullet points, concrete wording)
3. Overall fit verdict in one line (strong/moderate/weak match, why)

Be direct and specific. No generic advice."""

    response = llm.invoke(prompt)
    return response.content

def main():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    resume_text = extract_resume_text("resume.pdf")

    df = pd.read_csv("data/jobs_ranked.csv")
    top5 = df.head(5)

    results = []
    for _, row in top5.iterrows():
        print(f"Analyzing: {row['title']} at {row['company']}...")
        gap = analyze_gap(llm, resume_text, str(row["jd_text"]), row["title"], row["company"])
        results.append({
            "company": row["company"],
            "title": row["title"],
            "match_score": row["match_score"],
            "gap_analysis": gap
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv("data/gap_analysis.csv", index=False)
    print(f"\nSaved gap analysis for top {len(results)} matches to data/gap_analysis.csv")

if __name__ == "__main__":
    main()
