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

def draft_cover_note(llm, resume_text, jd_text, company_research, title, company):
    prompt = f"""Write a short, specific cover note (150-200 words) for this internship application.

RESUME:
{resume_text}

JOB: {title} at {company}

JOB DESCRIPTION:
{jd_text}

COMPANY RESEARCH:
{company_research}

Rules:
- Reference something specific from the company research (funding, product, recent news)
- Reference 1-2 specific resume achievements relevant to this JD
- No generic phrases like "I am passionate about" or "I am a quick learner"
- Direct, confident tone, no filler
"""
    response = llm.invoke(prompt)
    return response.content

def main():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)
    resume_text = extract_resume_text("resume.pdf")

    df = pd.read_csv("data/company_research.csv")

    drafts = []
    for _, row in df.iterrows():
        print(f"Drafting cover note: {row['title']} at {row['company']}...")
        draft = draft_cover_note(llm, resume_text, str(row["jd_text"]), row["company_research"], row["title"], row["company"])
        drafts.append(draft)

    df = df.copy()
    df["cover_note"] = drafts
    df.to_csv("data/cover_notes.csv", index=False)
    print(f"\nSaved {len(df)} cover notes to data/cover_notes.csv")

if __name__ == "__main__":
    main()
