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

    if _cached_time:
        return _cached_time
    return "Current time in India: unavailable"

# ============= TOOLS =============
@tool
def get_time(timezone: str = "Asia/Kolkata") -> str:
    """Get current time in a timezone. Use only when user asks for time."""
    return get_india_time()

@tool
def get_weather(location: str) -> str:
    """Get current weather for a city. Use only for weather/location queries."""
    return "Weather in Mumbai: 28°C, partly cloudy"  # Replace with real API later

@tool
def search_news(query: str) -> str:
    """Search latest news. Use only for news/current events questions."""
    return "Latest news: India wins cricket match!"

tools = [get_time, get_weather, search_news]

# ============= LLM =============
def get_llm(api_key: str, provider: str):
    if provider == "groq":
        return ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.7)
    return ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.7)

# ============= PROMPT (Fixed: Now has {tool_names} and {tools}) =============
REACT_PROMPT = PromptTemplate.from_template(
    """You are a helpful SMS assistant. Answer naturally and concisely.

Available tools: {tool_names}
{tools}

Use tools ONLY when needed:
- get_time → for time queries
- get_weather → for weather queries
- search_news → for news queries

Thought: Always think step by step. If no tool is needed, just respond.

Format:
Thought: [reasoning]
Action: tool_name
Action Input: {{"param": "value"}}
Observation: [tool result]
Final Answer: [your reply]

Question: {input}
{agent_scratchpad}"""
)

# ============= AGENT (With parsing error handling) =============
def run_agent_with_tools(conversation_history: list, latest_message: str, api_key: str, provider: str = "groq"):
    llm = get_llm(api_key, provider)
    
    agent = create_react_agent(llm, tools, REACT_PROMPT)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=3,
        verbose=False
    )
    
    try:
        response = executor.invoke({
            "input": latest_message,
            "chat_history": [HumanMessage(content=m["content"]) for m in conversation_history if m.get("content")]
        })
        return response["output"]
    except Exception as e:
        return f"Agent error: {str(e)}"

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

# ============= /int EXECUTION (Auto-run) =============
if "inputs" in globals():
    try:
        data = globals().get("inputs", {})
        conversation = data.get("conversation", [])
        message = data.get("message", "")
        api_key = data.get("api_key", "")
        provider = data.get("provider", "groq").lower()
        
        if not message or not api_key:
            result = {"error": "Missing message or api_key"}
        else:
            reply = run_agent_with_tools(conversation, message, api_key, provider)
            result = {
                "reply": reply,
                "india_time_used": get_india_time(),
                "status": "success",
                "provider": provider
            }
        
        globals()["result"] = result
    except Exception as e:
        globals()["result"] = {"error": str(e), "traceback": traceback.format_exc()}
