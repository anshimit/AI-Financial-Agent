import os
from dotenv import load_dotenv
import streamlit as st
from agent_engine import create_financial_agent
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, BaseMessage

# --- 1. INITIAL SETUP ---
load_dotenv()

# Force environment variables for underlying libraries
os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY") 
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")

st.set_page_config(page_title="Fin-AI Dashboard", layout="wide")

# --- 2. CACHED TOOLS & AGENT ---
@st.cache_resource
def get_rag_tool():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE")
    )
    db = Chroma(
        collection_name="AI_Initiatives", 
        embedding_function=embeddings, 
        persist_directory="./chroma_db"
    )

    retriever = db.as_retriever(search_kwargs={"k": 5})
        
    @tool
    def query_private_database(query: str) -> str:
        """
        Use this tool to search the internal research database. 
        It contains private PDF documents regarding company AI initiatives, 
        future projections, internal strategy papers, and research roadmaps.
        """
        docs = retriever.invoke(query)
        # Store sources in session state for the UI expander
        source_content = [d.page_content for d in docs]
        st.session_state.last_sources = source_content 
        
        if not source_content:
            return "No specific internal documents found for this query."
            
        return "\n\n".join(source_content)
    
# 4. CRITICAL: Return the tool so agent_engine can use it
    return query_private_database

# Initialize the tool and agent once
rag_tool = get_rag_tool()
agent = create_financial_agent(rag_tool)

# --- 3. SESSION STATE (MEMORY) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = "analysis"

# --- 4. MAIN UI ---
st.title("📈 Financial Intelligence Dashboard")
st.markdown("Ask about stock trends or internal AI research from your database.")

# Display existing chat history
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# --- 5. CHAT INPUT & EXECUTION ---
if prompt := st.chat_input("Analyze MSFT..."):
    # Save ticker context for the download filename
    st.session_state.current_ticker = prompt.replace(" ", "_")
    
    # Add User Message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.status("Agent researching...", expanded=False) as status:
            st.write("Accessing Financial Tools...")
            
            # Run Agent with History
            inputs = {"messages": st.session_state.messages}
            result = agent.invoke(inputs)
            
            # Extract final message content
            response_content = result["messages"][-1].content
            status.update(label="Analysis Complete!", state="complete")
        
        st.markdown(response_content)

        # --- NEW: SOURCE TRANSPARENCY EXPANDER ---
        if st.session_state.last_sources:
            with st.expander("🔍 View Internal Research Sources"):
                for i, source in enumerate(st.session_state.last_sources):
                    st.markdown(f"**Source {i+1}:**")
                    st.caption(source)
                    st.divider()

        st.session_state.messages.append(AIMessage(content=response_content))
        st.rerun()

# --- 6. SIDEBAR (Placed at end to capture updated state) ---
with st.sidebar:
    st.title("⚙️ Settings & Status")
    st.success("API Connected (GL Proxy)")
    st.info("Model: gpt-4o-mini")
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.current_ticker = "analysis"
        st.rerun()
        
    st.divider() 
    st.subheader("📄 Export Analysis")
    
    if st.session_state.messages:
        # Prepare the transcript text
        transcript = "FINANCIAL INTELLIGENCE REPORT\n" + "="*30 + "\n\n"
        for msg in st.session_state.messages:
            role = "USER" if isinstance(msg, HumanMessage) else "AGENT"
            transcript += f"{role}:\n{msg.content}\n\n"
        
        # Safe filename using session state
        file_name = f"report_{st.session_state.current_ticker}.txt"
        
        st.download_button(
            label="Download Research Brief",
            data=transcript,
            file_name=file_name,
            mime="text/plain"
        )
    else:
        st.caption("Start an analysis to enable downloads.")