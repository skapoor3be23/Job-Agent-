import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import re

load_dotenv()

CRITIQUE_PROMPT = """Score this cover note against the job description on a scale of 1-10.

JOB DESCRIPTION:
{jd_text}

COVER NOTE:
{cover_note}

Return in this exact format:
SCORE: <number>
ISSUES: <max 3 specific issues, or "None" if score is 8+>
"""

REGENERATE_PROMPT = """Rewrite this cover note to fix these specific issues: {issues}

JOB DESCRIPTION:
{jd_text}

ORIGINAL COVER NOTE:
{cover_note}

Rules: 150-200 words, no generic phrases, direct tone. Return only the rewritten note."""

def critique(llm, jd_text, cover_note):
    prompt = CRITIQUE_PROMPT.format(jd_text=jd_text, cover_note=cover_note)
    response = llm.invoke(prompt).content
    score_match = re.search(r"SCORE:\s*(\d+)", response)
    issues_match = re.search(r"ISSUES:\s*(.+)", response, re.DOTALL)
    score = int(score_match.group(1)) if score_match else 5
    issues = issues_match.group(1).strip() if issues_match else "Unknown"
    return score, issues

def regenerate(llm, jd_text, cover_note, issues):
    prompt = REGENERATE_PROMPT.format(issues=issues, jd_text=jd_text, cover_note=cover_note)
    return llm.invoke(prompt).content

def main():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    df = pd.read_csv("data/cover_notes.csv")

    final_notes = []
    scores = []

    for _, row in df.iterrows():
        print(f"Critiquing: {row['title']} at {row['company']}...")
        score, issues = critique(llm, str(row["jd_text"]), row["cover_note"])

        if score < 8:
            print(f"  Score {score}/10, issues: {issues}. Regenerating...")
            new_note = regenerate(llm, str(row["jd_text"]), row["cover_note"], issues)
            final_notes.append(new_note)
            scores.append(score)
        else:
            print(f"  Score {score}/10. Keeping original.")
            final_notes.append(row["cover_note"])
            scores.append(score)

    df = df.copy()
    df["initial_score"] = scores
    df["final_cover_note"] = final_notes
    df.to_csv("data/final_cover_notes.csv", index=False)
    print(f"\nSaved final cover notes to data/final_cover_notes.csv")

if __name__ == "__main__":
    main()
