import sys
from openai import OpenAI
from src.config import OPENAI_API_KEY, DEFAULT_MODEL
from src.state import load_state
from src.coa import coa_report

client = OpenAI(api_key=OPENAI_API_KEY)
state = load_state()

question = " ".join(sys.argv[1:]).strip() or "Summarize the case: timeline, key findings, conflicts, gaps."
report = coa_report(client, DEFAULT_MODEL, state["vector_store_id"], question, n_workers=4)

with open("report.md", "w", encoding="utf-8") as f:
    f.write(report)

print("Wrote report.md")