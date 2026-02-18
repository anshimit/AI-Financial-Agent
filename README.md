# AI-Financial-Agent
This project was built to experiment with Agentic RAG applications and create an agent that can autonomously provide market research analysis using real-time market data and proprietary research 

### Why this architecture?

State Machines vs. Linear Chains: Instead of a basic RetrievalQA chain, I used LangGraph. This allows the agent to "loop"—if it retrieves a stock price and realizes it needs more context from an internal PDF to explain a trend, it can trigger a second tool-call automatically.

Source Transparency: I implemented a custom "Source Expander" in Streamlit. This solves the "black box" problem of RAG by showing the exact text chunks from the PDFs that influenced the agent's answer.

Hybrid Data Access: The agent manages two distinct tools:

yFinance API: For high-fidelity, real-time market numbers.

ChromaDB Vector Store: For semantic search across internal PDF research.

### The Tech Stack

Orchestration: LangGraph (StateGraph for cycle management)

LLM: GPT-4o-mini

Vector Database: ChromaDB (Local persistent storage)

UI: Streamlit

Libraries: LangChain, yFinance, PyPDF2

### Getting Started

1. Clone & Environment

2. Secrets Configuration
Create a .env file in the root directory. You'll need an OpenAI API key.

3. Run the Dashboard

### Challenges I Solved

Message Reduction: Handling long PDF chunks in the chat history can quickly hit token limits. I implemented a system to filter and condense tool outputs before they are passed back to the AgentState.

Tool Routing: Fine-tuning the tool descriptions so the LLM correctly distinguishes between "Current Market Price" (API) and "Internal Strategy/Risk" (RAG).

### Future Roadmap

[ ] Transition from local ChromaDB to a cloud-based vector store (Pinecone).

[ ] Add an "Audit Trail" tab to see the raw JSON exchange between the Agent and the Tools.
