"""
Live, interactive version of pipeline.py for the Streamlit dashboard.
Same LangGraph structure and prompts as pipeline.py, but:
- Takes resume text + JD text as direct inputs (no local resume.pdf / CSV)
- Reads API keys from st.secrets instead of .env
- Exposes progress callbacks so the UI can show which node is running
"""

import re
import streamlit as st
from typing import TypedDict, Callable, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from langgraph.graph import StateGraph, END


class JobState(TypedDict):
    resume_text: str
    company: str
    title: str
    jd_text: str
    gap_analysis: str
    company_research: str
    cover_note: str
    critique_score: int
    critique_issues: str
    final_cover_note: str


def _get_clients():
    """Build LLM + Tavily clients from Streamlit secrets. Raises a clear error if missing."""
    try:
        google_key = st.secrets["GOOGLE_API_KEY"]
        tavily_key = st.secrets["TAVILY_API_KEY"]
    except (KeyError, FileNotFoundError):
        raise RuntimeError(
            "Missing API keys. Add GOOGLE_API_KEY and TAVILY_API_KEY in "
            "Streamlit Cloud → App settings → Secrets."
        )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0.3, google_api_key=google_key
    )
    tavily = TavilyClient(api_key=tavily_key)
    return llm, tavily


def _build_graph(llm, tavily):
    def node_gap_analysis(state: JobState) -> JobState:
        if not state["jd_text"] or len(state["jd_text"].strip()) < 20:
            state["gap_analysis"] = "SKIPPED: job description empty or too short."
            return state
        try:
            prompt = f"""Compare resume against job description.
RESUME: {state['resume_text']}
JOB ({state['title']} at {state['company']}): {state['jd_text']}
Return: 1) Missing skills (max 8) 2) Resume edits (max 3) 3) Fit verdict (one line)."""
            state["gap_analysis"] = llm.invoke(prompt).content
        except Exception as e:
            state["gap_analysis"] = f"ERROR: {str(e)}"
        return state

    def node_company_research(state: JobState) -> JobState:
        try:
            query = f"{state['company']} company funding size recent news"
            response = tavily.search(query, max_results=3)
            raw = ""
            for r in response.get("results", []):
                raw += f"- {r.get('title', '')}: {r.get('content', '')[:400]}\n"
            if not raw:
                state["company_research"] = "No information found."
                return state
            # Synthesize raw search fragments into clean prose instead of dumping them
            synth_prompt = f"""Below are raw web search snippets about {state['company']}.
Write a clean 3-4 sentence summary covering: what the company does, size/scale if mentioned,
and one recent relevant fact. Do not include broken sentences, metadata labels, or bullet fragments.
If the snippets are too fragmented to summarize reliably, say so plainly instead of guessing.

RAW SNIPPETS:
{raw}"""
            state["company_research"] = llm.invoke(synth_prompt).content
        except Exception as e:
            state["company_research"] = f"ERROR: {str(e)}"
        return state

    def node_draft_cover_note(state: JobState) -> JobState:
        if state["gap_analysis"].startswith("SKIPPED") or state["gap_analysis"].startswith("ERROR"):
            state["cover_note"] = "SKIPPED: upstream stage failed."
            return state
        try:
            prompt = f"""Write a 150-200 word cover note.
RESUME: {state['resume_text']}
JOB: {state['title']} at {state['company']}
JD: {state['jd_text']}
COMPANY RESEARCH: {state['company_research']}
Rules: reference specific company facts, reference specific resume achievements, no generic phrases."""
            state["cover_note"] = llm.invoke(prompt).content
        except Exception as e:
            state["cover_note"] = f"ERROR: {str(e)}"
        return state

    def node_critique(state: JobState) -> JobState:
        if state["cover_note"].startswith("SKIPPED") or state["cover_note"].startswith("ERROR"):
            state["critique_score"] = 0
            state["critique_issues"] = "Skipped due to upstream failure."
            return state
        try:
            prompt = f"""You are a strict, skeptical hiring manager reviewing a cover note.
Be harsh — most cover notes have real flaws. Do not default to high scores.

JD: {state['jd_text']}
NOTE: {state['cover_note']}

Score on this rubric (each out of 2, sum to get total /10):
1. Specificity: does it cite concrete, verifiable details from the resume (not vague claims)?
2. JD alignment: does it address the JD's actual stated requirements, not just adjacent skills?
3. Company relevance: does it use a genuine, specific company fact (not generic filler)?
4. No generic phrases: no "passionate," "quick learner," "team player," or similar filler?
5. Would a human reviewer keep reading past the first two lines?

For each gap you notice in the ISSUES list, you MUST deduct at least 1 point total.
A score of 8+ requires zero notable issues. If you list any issues, score must be 7 or below.

Return exactly:
SCORE: <number>
ISSUES: <specific issues found, or None only if score is 8+>"""
            response = llm.invoke(prompt).content
            score_match = re.search(r"SCORE:\s*(\d+)", response)
            issues_match = re.search(r"ISSUES:\s*(.+)", response, re.DOTALL)
            state["critique_score"] = int(score_match.group(1)) if score_match else 5
            state["critique_issues"] = issues_match.group(1).strip() if issues_match else "Unknown"
        except Exception as e:
            state["critique_score"] = 0
            state["critique_issues"] = f"ERROR: {str(e)}"
        return state

    def node_regenerate(state: JobState) -> JobState:
        if state["cover_note"].startswith("SKIPPED") or state["cover_note"].startswith("ERROR"):
            state["final_cover_note"] = state["cover_note"]
            return state
        if 0 < state["critique_score"] < 8:
            try:
                prompt = f"""Rewrite this cover note fixing: {state['critique_issues']}
JD: {state['jd_text']}
ORIGINAL: {state['cover_note']}
Rules: 150-200 words, no generic phrases, direct tone. Return only the note."""
                state["final_cover_note"] = llm.invoke(prompt).content
            except Exception as e:
                state["final_cover_note"] = state["cover_note"]
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

    return graph.compile()


def run_live_pipeline(
    resume_text: str,
    company: str,
    title: str,
    jd_text: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> JobState:
    """
    Run the full 5-node pipeline live and return the final state.
    progress_callback(msg) is called before each stage if provided,
    so the caller (Streamlit UI) can show a spinner/status text.
    """
    llm, tavily = _get_clients()
    pipeline = _build_graph(llm, tavily)

    initial_state: JobState = {
        "resume_text": resume_text,
        "company": company,
        "title": title,
        "jd_text": jd_text,
        "gap_analysis": "",
        "company_research": "",
        "cover_note": "",
        "critique_score": 0,
        "critique_issues": "",
        "final_cover_note": "",
    }

    if progress_callback:
        progress_callback("Running gap analysis, company research, cover note, critique, and rewrite...")

    return pipeline.invoke(initial_state)
