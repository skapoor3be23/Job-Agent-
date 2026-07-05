# Job Application Agent

An agentic pipeline that automates internship/job application prep: matches your resume against job listings, identifies skill gaps, researches target companies, drafts tailored cover notes, and self-critiques/improves them before you apply.

## Pipeline stages

1. **Ingestion** (`fetch_jobs.py`) — pulls job listings via Adzuna API
2. **Matching** (`match_jobs.py`) — embeds resume + JDs (Gemini embeddings), ranks by cosine similarity
3. **Gap analysis** (`gap_analysis.py`) — LLM identifies missing skills and suggests resume edits
4. **Company research** (`company_research.py`) — Tavily web search for funding/size/news
5. **Drafting** (`draft_cover_notes.py`) — LLM writes cover note using resume + JD + company research
6. **Critique** (`critique_notes.py`) — second LLM pass scores the draft, regenerates if score < 8
7. **Pipeline** (`pipeline.py`) — wires stages 2-6 into a single LangGraph state machine
8. **Tracking** (`tracker.py`) — SQLite DB with dedup, status tracking (applied/interview/rejected)
9. **Dashboard** (`dashboard.py`) — Streamlit UI for viewing matches, updating status, reading cover notes

## Stack
LangChain, LangGraph, Gemini 2.5 Flash, Gemini embeddings, FAISS-style cosine similarity, Tavily search API, Adzuna API, SQLite, Streamlit.

## Setup
1. `pip install -r requirements.txt`
2. Add `.env` with `GOOGLE_API_KEY`, `TAVILY_API_KEY`, `ADZUNA_APP_ID`, `ADZUNA_APP_KEY`
3. Place your resume as `resume.pdf` in root
4. Run stages in order, or run `pipeline.py` directly for the full flow

## Known limitations
- Free-tier Gemini API caps at 20 requests/day, limiting batch size per run
- Feedback/correlation loop (matching resume version to response rate) not yet implemented — requires real outcome data over time
- No error handling for malformed/empty job descriptions
