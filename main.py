import os
import json
from pathlib import Path

import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.company_ranking import rank_companies


def load_config():
    root = Path(__file__).parent
    config_path = root / "config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
    if "OPENAI_API_BASE" in config:
        os.environ["OPENAI_BASE_URL"] = config["OPENAI_API_BASE"]


def main():
    root = Path(__file__).parent

    # 1) Load API keys
    load_config()

    # 2) Load financial data
    df = pd.read_csv(root / "data" / "financial_data.csv")

    # 3) Load vectorstore
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name="AI_Initiatives",
        embedding_function=embeddings,
        persist_directory=str(root / "chroma_db"),
    )

    # 4) LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=2000,
    )

    # 5) Call ranking function (✅ THIS MUST BE HERE)
    result = rank_companies(
        df=df,
        vectorstore=vectorstore,
        llm=llm
    )

    print(result)


# ✅ This is what actually runs your program
if __name__ == "__main__":
    main()
# End of main.py