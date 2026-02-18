import os
from typing import Dict, List, Literal, Annotated, Sequence
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import yfinance as yf

load_dotenv()

# --- TOOLS ---
@tool
def get_stock_price(ticker: str) -> Dict:
    """Fetch real-time price, volume, and market cap for a given stock ticker."""
    stock = yf.Ticker(ticker.upper())
    info = stock.info
    return {
        "ticker": ticker.upper(),
        "price": info.get('currentPrice'),
        "market_cap": info.get('marketCap'),
        "timestamp": datetime.now().isoformat()
    }

@tool
def get_stock_history(ticker: str, period: str = "3y") -> Dict:
    """Fetch historical returns and performance trends over a specific period (default 3 years)."""
    stock = yf.Ticker(ticker.upper())
    hist = stock.history(period=period)
    if hist.empty: return {"error": "No data"}
    return {"return_pct": round(((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100, 2)}

# --- AGENT SETUP ---
class AgentState(MessagesState):
    # This tells LangGraph exactly how to handle the message list
    messages: Annotated[Sequence[BaseMessage], add_messages]

def create_financial_agent(retriever_tool):
    # Ensure all three tools are present
    tools = [get_stock_price, get_stock_history, retriever_tool]
    
    model = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0,
        openai_api_key=os.getenv("API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE") 
    ).bind_tools(tools)

    def agent_node(state: AgentState):
        # --- ENHANCED SYSTEM MESSAGE ---
        # We explicitly tell the agent to use the private database for internal info.
        system_msg = SystemMessage(content=(
            "You are an expert Financial Analyst with access to real-time market data AND a private database of internal company AI initiatives. "
            "For any question about internal strategy, research, or AI projections, you MUST use the 'query_private_database' tool. "
            "Combine market data and internal research to provide a comprehensive recommendation."
        ))
        
        # Invoke the model with the system message + full conversation history
        response = model.invoke([system_msg] + state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        # Check if the model wants to use a tool
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return "end"

    # --- GRAPH CONSTRUCTION ---
    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools)) # ToolNode automatically handles the tool list
    
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent", 
        should_continue, 
        {
            "tools": "tools", 
            "end": END
        }
    )
    
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()