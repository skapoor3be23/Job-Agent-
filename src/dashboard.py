import streamlit as st
import pandas as pd
import sqlite3

st.set_page_config(page_title="Job Application Agent", layout="wide")
st.title("Job Application Agent Dashboard")

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
