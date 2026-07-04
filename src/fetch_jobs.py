import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

APP_ID = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")

COUNTRY = "in"  # India
QUERY = "AI ML intern"
RESULTS_PER_PAGE = 50

def fetch_jobs():
    url = f"https://api.adzuna.com/v1/api/jobs/{COUNTRY}/search/1"
    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": RESULTS_PER_PAGE,
        "what": QUERY,
        "content-type": "application/json"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    jobs = []
    for job in data.get("results", []):
        jobs.append({
            "company": job.get("company", {}).get("display_name", ""),
            "title": job.get("title", ""),
            "jd_text": job.get("description", ""),
            "location": job.get("location", {}).get("display_name", ""),
            "link": job.get("redirect_url", "")
        })

    df = pd.DataFrame(jobs)
    df.to_csv("data/jobs.csv", index=False)
    print(f"Saved {len(df)} jobs to data/jobs.csv")

if __name__ == "__main__":
    fetch_jobs()
