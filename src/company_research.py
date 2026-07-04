import os
import pandas as pd
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def research_company(company_name):
    query = f"{company_name} company funding size recent news"
    response = client.search(query, max_results=3)
    summary = ""
    for r in response.get("results", []):
        summary += f"- {r.get('title', '')}: {r.get('content', '')[:200]}\n"
    return summary if summary else "No information found."

def main():
    df = pd.read_csv("data/jobs_ranked.csv")
    top5 = df.head(5)

    research_notes = []
    for _, row in top5.iterrows():
        print(f"Researching: {row['company']}...")
        notes = research_company(row["company"])
        research_notes.append(notes)

    top5 = top5.copy()
    top5["company_research"] = research_notes
    top5.to_csv("data/company_research.csv", index=False)
    print(f"\nSaved company research for {len(top5)} companies to data/company_research.csv")

if __name__ == "__main__":
    main()
