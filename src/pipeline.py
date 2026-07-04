import os
import pandas as pd
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from tavily import TavilyClient
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict
import numpy as np
import re

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

class JobState(TypedDict):
    resume_text: str
    company: str
    title: str
    jd_text: str
    match_score: float
    gap_analysis: str
    company_research: str
    cover_note: str
    critique_score: int
    critique_issues: str
    final_cover_note: str

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def node_gap_analysis(state: JobState) -> JobState:
    prompt = f"""Compare resume against job description.
RESUME: {state['resume_text']}
JOB ({state['title']} at {state['company']}): {state['jd_text']}
Return: 1) Missing skills (max 8) 2) Resume edits (max 3) 3) Fit verdict (one line)."""
    state["gap_analysis"] = llm.invoke(prompt).content
    return state

def node_company_research(state: JobState) -> JobState:
    query = f"{state['company']} company funding size recent news"
    response = tavily.search(query, max_results=3)
    summary = ""
    for r in response.get("results", []):
        summary += f"- {r.get('title','')}: {r.get('content','')[:200]}\n"
    state["company_research"] = summary if summary else "No information found."
    return state

def node_draft_cover_note(state: JobState) -> JobState:
    prompt = f"""Write a 150-200 word cover note.
RESUME: {state['resume_text']}
JOB: {state['title']} at {state['company']}
JD: {state['jd_text']}
COMPANY RESEARCH: {state['company_research']}
Rules: reference specific company facts, reference specific resume achievements, no generic phrases."""
    state["cover_note"] = llm.invoke(prompt).content
    return state

def node_critique(state: JobState) -> JobState:
    prompt = f"""Score this cover note 1-10 against the JD.
JD: {state['jd_text']}
NOTE: {state['cover_note']}
Return: SCORE: <number>\nISSUES: <max 3 issues or None>"""
    response = llm.invoke(prompt).content
    score_match = re.search(r"SCORE:\s*(\d+)", response)
    issues_match = re.search(r"ISSUES:\s*(.+)", response, re.DOTALL)
    state["critique_score"] = int(score_match.group(1)) if score_match else 5
    state["critique_issues"] = issues_match.group(1).strip() if issues_match else "Unknown"
    return state

def node_regenerate(state: JobState) -> JobState:
    if state["critique_score"] < 8:
        prompt = f"""Rewrite this cover note fixing: {state['critique_issues']}
JD: {state['jd_text']}
ORIGINAL: {state['cover_note']}
Rules: 150-200 words, no generic phrases, direct tone. Return only the note."""
        state["final_cover_note"] = llm.invoke(prompt).content
    else:
        state["final_cover_note"] = state["cover_note"]
    return state

graph = StateGraph(JobState)
graph.add_node("gap_analysis", node_gap_analysis)
graph.add_node("company_research", node_company_research)
graph.add_node("draft_cover_note", node_draft_cover_note)
graph.add_node("critique", node_critique)
graph.add_node("regenerate", node_regenerate)

graph.set_entry_point("gap_analysis")
graph.add_edge("gap_analysis", "company_research")
graph.add_edge("company_research", "draft_cover_note")
graph.add_edge("draft_cover_note", "critique")
graph.add_edge("critique", "regenerate")
graph.add_edge("regenerate", END)

pipeline = graph.compile()

def extract_resume_text(path="resume.pdf"):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def main():
    resume_text = extract_resume_text("resume.pdf")
    df = pd.read_csv("data/jobs_ranked.csv")
    top5 = df.head(1)

    results = []
    for _, row in top5.iterrows():
        print(f"Running pipeline: {row['title']} at {row['company']}...")
        initial_state: JobState = {
            "resume_text": resume_text,
            "company": row["company"],
            "title": row["title"],
            "jd_text": str(row["jd_text"]),
            "match_score": row["match_score"],
            "gap_analysis": "",
            "company_research": "",
            "cover_note": "",
            "critique_score": 0,
            "critique_issues": "",
            "final_cover_note": ""
        }
        final_state = pipeline.invoke(initial_state)
        results.append({
            "company": final_state["company"],
            "title": final_state["title"],
            "match_score": final_state["match_score"],
            "gap_analysis": final_state["gap_analysis"],
            "company_research": final_state["company_research"],
            "critique_score": final_state["critique_score"],
            "final_cover_note": final_state["final_cover_note"]
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv("data/pipeline_output.csv", index=False)
    print(f"\nPipeline complete. Saved {len(out_df)} results to data/pipeline_output.csv")

if __name__ == "__main__":
    main()
