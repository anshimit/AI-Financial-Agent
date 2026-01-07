"""
Company Ranking Pipeline
------------------------
Scores and ranks companies using:
1) Financial data
2) AI initiative evidence from a vectorstore
"""

def build_ai_evidence_pack(vectorstore, max_docs=50, max_chars=600):
    """
    Build a compact evidence pack from vectorstore documents
    to avoid context-length overflow.
    """
    docs = vectorstore.get()["documents"]
    docs_small = [d[:max_chars] for d in docs[:max_docs]]
    return "\n\n".join(f"- {d}" for d in docs_small)


def rank_companies(df, vectorstore, llm):
    """
    Rank companies using financial data + AI initiative evidence.
    """
    ai_evidence_pack = build_ai_evidence_pack(vectorstore)

    system_message = """
You are an investment and technology analyst. Your task is to SCORE and RANK companies using ONLY the data provided.

Scoring rubric (0–100):
- Financial Strength (0–40)
- AI Initiative Strength (0–40)
- Strategic Fit (0–20)
"""

    user_message = f"""
Rank the companies using ONLY the data below.

---
### Financial Data
{df.to_string(index=False)}

---
### AI Initiatives (Evidence Pack)
{ai_evidence_pack}
"""

    prompt = f"""[INST]{system_message}

user: {user_message}
[/INST]"""

    return llm.invoke(prompt).content
