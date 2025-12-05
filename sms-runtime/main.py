# main.py - Real-time SMS AI Runtime (called from app.py via /int OR run as server)
import os
import json
import requests
import traceback
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

# LangChain imports
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# ============= CACHED INDIA TIME (Never fails) =============
_cached_time = None
_cached_at = 0

def get_india_time():
    global _cached_time, _cached_at
    now = time.time()
    
    # Return cached if < 60 seconds old
    if _cached_time and (now - _cached_at) < 60:
        return _cached_time

    urls = [
        "https://worldtimeapi.org/api/timezone/Asia/Kolkata",
        "http://worldtimeapi.org/api/timezone/Asia/Kolkata",
        "https://timeapi.io/api/Time/current/zone?timeZone=Asia/Kolkata"
    ]

    for url in urls:
        try:
            resp = requests.get(url, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                dt = data.get("datetime") or data.get("dateTime")
                if dt:
                    time_str = dt.split("T")[1][:8] if "T" in dt else dt.split(" ")[1][:8]
                    result = f"Current time in India (Kolkata): {time_str}"
                    _cached_time = result
                    _cached_at = now
                    return result
        except:
            continue

    # Final fallback
    if _cached_time:
        return _cached_time
    return "Current time in India: unavailable"

# ============= TOOLS =============
@tool
def get_time(timezone: str = "Asia/Kolkata") -> str:
    """Get current time in any timezone. Use for time queries."""
    return get_india_time() if timezone == "Asia/Kolkata" else "Time tool used"

@tool
def get_weather(location: str) -> str:
    """Get weather for a city. Use only for weather queries."""
    return "Weather in Mumbai: 28°C, partly cloudy"  # Replace with real API later

@tool
def search_news(query: str) -> str:
    """Search news. Use only for news queries."""
    return "Latest news: India wins cricket match!"

tools = [get_time, get_weather, search_news]

# ============= LLM =============
def get_llm(api_key: str, provider: str):
    if provider == "groq":
        return ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.7)
    return ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.7)

# ============= PROMPT (Fixed for parsing) =============
REACT_PROMPT = PromptTemplate.from_template("""
You are a helpful SMS assistant.

Tools: {tool_names}

Use tools only when needed:
- get_time: for time queries
- get_weather: for weather
- search_news: for news

Thought: Reason step by step. If no tool needed, just respond.

Format:
Thought: [your reasoning]
Action: tool_name
Action Input: {{"arg": "value"}}
Observation: [result]
Final Answer: [reply to user]

Question: {input}
{agent_scratchpad}
""")

# ============= AGENT (With parsing error handling) =============
def run_agent_with_tools(conversation_history: list, latest_message: str, api_key: str, provider: str = "groq"):
    llm = get_llm(api_key, provider)
    
    agent = create_react_agent(llm, tools, REACT_PROMPT)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,  # ← THIS FIXES THE PARSING ERROR
        max_iterations=3,
        verbose=False
    )
    
    try:
        response = executor.invoke({
            "input": latest_message,
            "chat_history": [HumanMessage(content=m["content"]) for m in conversation_history]
        })
        return response["output"]
    except Exception as e:
        return f"Agent failed: {str(e)}"

# ============= FASTAPI =============
app = FastAPI()

@app.post("/run-agent")
async def run_agent(request: Request):
    try:
        body = await request.json()
    except:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    
    conversation = body.get("conversation", [])
    message = body.get("message", "").strip()
    api_key = body.get("api_key", "").strip()
    provider = body.get("provider", "groq").lower()
    
    if not message or not api_key or provider not in ["groq", "openai"]:
        return JSONResponse({"error": "Invalid request"}, status_code=400)
    
    reply = run_agent_with_tools(conversation, message, api_key, provider)
    
    return JSONResponse({
        "reply": reply,
        "india_time_used": get_india_time(),
        "status": "success",
        "provider": provider
    })

@app.get("/")
def health():
    return {"status": "AI SMS Runtime LIVE", "time": get_india_time()}

# ============= /int EXECUTION =============
if "inputs" in globals():
    try:
        data = globals().get("inputs", {})
        conversation = data.get("conversation", [])
        message = data.get("message", "")
        api_key = data.get("api_key", "")
        provider = data.get("provider", "groq").lower()
        
        if not message or not api_key:
            result = {"error": "Missing data"}
        else:
            reply = run_agent_with_tools(conversation, message, api_key, provider)
            result = {
                "reply": reply,
                "india_time_used": get_india_time(),
                "status": "success"
            }
        
        globals()["result"] = result
    except Exception as e:
        globals()["result"] = {"error": str(e)}
