import os
from dotenv import load_dotenv
import streamlit as st
import yfinance as yf
import requests

load_dotenv()

def verify():
    print("--- Environment Check ---")
    
    # 1. Check API Key
    key = os.getenv("API_KEY")
    base = os.getenv("OPENAI_API_BASE")
    
    if key and key.startswith("gl-"):
        print(f"✅ API Key: Found ({key[:6]}...)")
    else:
        print("❌ API Key: Missing or doesn't start with 'gl-'")

    # 2. Check API Base (Great Learning Proxy)
    if base == "https://aibe.mygreatlearning.com/openai/v1":
        print(f"✅ API Base: Correctly set to Great Learning Proxy")
    else:
        print(f"⚠️  API Base: Unexpected value ({base})")

    # 3. YFinance check
    try:
        data = yf.Ticker("AAPL").info
        print(f"✅ YFinance: Working (Fetched {data.get('shortName')})")
    except Exception as e:
        print(f"❌ YFinance: Failed ({e})")

if __name__ == "__main__":
    verify()