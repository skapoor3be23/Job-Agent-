import sqlite3
import pandas as pd

def init_db():
    conn = sqlite3.connect("data/tracker.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT,
            title TEXT,
            match_score REAL,
            status TEXT DEFAULT 'not_applied',
            date_applied TEXT,
            cover_note TEXT
        )
    """)
    conn.commit()
    return conn

def load_pipeline_output(conn):
    df = pd.read_csv("data/pipeline_output.csv")
    for _, row in df.iterrows():
        existing = conn.execute(
            "SELECT id FROM applications WHERE LOWER(TRIM(company)) = LOWER(?) "
            "AND LOWER(TRIM(title)) = LOWER(?)",
            (str(row["company"]).strip(), str(row["title"]).strip())
        ).fetchone()
        if existing:
            continue
        conn.execute(
            "INSERT INTO applications (company, title, match_score, cover_note) VALUES (?, ?, ?, ?)",
            (row["company"], row["title"], row["match_score"], row["final_cover_note"])
        )
    conn.commit()

def view_all(conn):
    df = pd.read_sql("SELECT id, company, title, match_score, status, date_applied FROM applications", conn)
    print(df)

def update_status(conn, app_id, new_status, date=None):
    conn.execute(
        "UPDATE applications SET status = ?, date_applied = ? WHERE id = ?",
        (new_status, date, app_id)
    )
    conn.commit()

def correlation_report(conn):
    df = pd.read_sql("SELECT status, COUNT(*) as count FROM applications GROUP BY status", conn)
    print("\nStatus breakdown:")
    print(df)

if __name__ == "__main__":
    conn = init_db()
    load_pipeline_output(conn)
    view_all(conn)
    conn.close()
    print("\nSaved to data/tracker.db")