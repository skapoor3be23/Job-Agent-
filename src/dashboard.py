import streamlit as st
import pandas as pd
import sqlite3
from pypdf import PdfReader
from live_pipeline import run_live_pipeline

st.set_page_config(page_title="Job Application Agent", layout="wide")
st.title("Job Application Agent Dashboard")

st.subheader("Run Pipeline Live")
st.caption(
    "Upload your resume and paste a job description to run the full "
    "5-agent pipeline (gap analysis → company research → cover note → "
    "critique → rewrite) live. Takes 15-30 seconds."
)

with st.form("live_run_form"):
    col1, col2 = st.columns(2)
    with col1:
        live_company = st.text_input("Company")
        live_title = st.text_input("Job title")
    with col2:
        resume_file = st.file_uploader("Resume (PDF)", type=["pdf"])

    jd_input_mode = st.radio("Job description input", ["Paste text", "Upload PDF"], horizontal=True)
    if jd_input_mode == "Paste text":
        live_jd = st.text_area("Job description", height=150)
        jd_file = None
    else:
        jd_file = st.file_uploader("Job description (PDF)", type=["pdf"], key="jd_pdf")
        live_jd = ""

    submitted = st.form_submit_button("Run pipeline")

if submitted:
    if not resume_file:
        st.error("Upload a resume PDF first.")
    elif jd_input_mode == "Paste text" and (not live_jd or len(live_jd.strip()) < 20):
        st.error("Paste a job description (at least a few sentences).")
    elif jd_input_mode == "Upload PDF" and not jd_file:
        st.error("Upload a job description PDF.")
    elif not live_company or not live_title:
        st.error("Fill in company and job title.")
    else:
        try:
            reader = PdfReader(resume_file)
            resume_text = "".join(page.extract_text() or "" for page in reader.pages)

            if jd_input_mode == "Upload PDF":
                jd_reader = PdfReader(jd_file)
                live_jd = "".join(page.extract_text() or "" for page in jd_reader.pages)
                if len(live_jd.strip()) < 20:
                    st.error("Could not extract readable text from the JD PDF.")
                    st.stop()

            with st.status("Running pipeline...", expanded=True) as status:
                st.write("Extracting resume text... done.")
                st.write("Running gap analysis, company research, cover note, critique, and rewrite...")
                result = run_live_pipeline(resume_text, live_company, live_title, live_jd)
                status.update(label="Pipeline complete", state="complete")

            st.subheader("Gap Analysis")
            st.write(result["gap_analysis"])

            st.subheader("Company Research")
            st.write(result["company_research"])

            st.subheader("Critique")
            st.write(f"Score: {result['critique_score']}/10 — {result['critique_issues']}")

            st.subheader("Final Cover Note")
            st.text_area("Result", result["final_cover_note"], height=250)

        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Pipeline failed: {e}")

st.divider()

conn = sqlite3.connect("data/tracker.db")
df = pd.read_sql("SELECT * FROM applications", conn)

st.subheader("Applications")
status_filter = st.selectbox("Filter by status", ["All"] + df["status"].unique().tolist())
if status_filter != "All":
    df_display = df[df["status"] == status_filter]
else:
    df_display = df

st.dataframe(df_display[["id", "company", "title", "match_score", "status", "date_applied"]])

st.subheader("Update Status")
app_id = st.selectbox("Select application ID", df["id"].tolist())
new_status = st.selectbox("New status", ["not_applied", "applied", "interview", "rejected", "ghosted"])
if st.button("Update"):
    conn.execute("UPDATE applications SET status = ? WHERE id = ?", (new_status, app_id))
    conn.commit()
    st.success(f"Updated ID {app_id} to {new_status}")
    st.rerun()

st.subheader("Cover Note")
selected_row = df[df["id"] == app_id]
if not selected_row.empty:
    st.text_area("Final cover note", selected_row.iloc[0]["cover_note"], height=300)

st.subheader("Status Breakdown")
status_counts = df["status"].value_counts()
st.bar_chart(status_counts)

conn.close()
