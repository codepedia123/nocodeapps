# main.py - Real-time SMS AI Runtime (called from app.py via /int OR run as server)
import os
import json
import requests
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

# LangChain imports for agent + tools
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# ============= CONFIG =============
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# ============= TOOL DEFINITIONS (API Requests with Descriptions for Conditional Triggering) =============
# These tools are evaluated by LangChain ReAct agent — LLM decides when to call based on query (case-based, not always)
# Descriptions guide triggering: e.g., time tool only for time queries, weather only for location/weather mentions

class TimeInput(BaseModel):
    timezone: str = Field(default="Asia/Kolkata", description="Timezone like 'Asia/Kolkata'")

@tool(args_schema=TimeInput)
def get_time(timezone: str) -> str:
    """Get current time in a timezone. Trigger only for time-related queries, e.g., 'What time is it in [place]?' or 'Current time in India?'"""
    try:
        resp = requests.get(f"https://worldtimeapi.org/api/timezone/{timezone}", timeout=5)
        data = resp.json()
        dt = data["datetime"]
        time_str = dt.split("T")[1][:8]
        return f"Current time in {timezone}: {time_str}"
    except Exception as e:
        return f"Time lookup failed: {str(e)}"

class WeatherInput(BaseModel):
    location: str = Field(description="City or location for weather, e.g., 'Mumbai'")

@tool(args_schema=WeatherInput)
def get_weather(location: str) -> str:
    """Get current weather for a location. Trigger only for weather/temperature queries in a specific place, e.g., 'Weather in Delhi?' or 'Temperature in Bangalore'."""
    url = "https://api.openweathermap.org/data/2.5/weather"  # Method: GET
    params = {"q": location, "appid": "your_openweather_key", "units": "metric"}  # Replace with your key
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("cod") == 200:
            return f"Weather in {location}: {data['weather'][0]['description']}, {data['main']['temp']}°C, feels like {data['main']['feels_like']}°C"
        return f"Weather data not found for {location}"
    except Exception as e:
        return f"Weather lookup failed: {str(e)}"

class NewsInput(BaseModel):
    query: str = Field(description="Search term for news, e.g., 'India election'")

@tool(args_schema=NewsInput)
def search_news(query: str) -> str:
    """Search recent news articles. Trigger only for news/current events questions like 'Latest news on [topic]' or 'What happened in India today?'."""
    url = "https://newsapi.org/v2/everything"  # Method: GET
    params = {"q": query, "apiKey": "your_newsapi_key", "sortBy": "publishedAt", "pageSize": 3}  # Replace with your key
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("status") == "ok" and data["articles"]:
            articles = data["articles"]
            summary = "\n".join([f"- {a['title']} ({a['publishedAt'][:10]})" for a in articles])
            return f"Latest news on '{query}':\n{summary}"
        return "No news found for query"
    except Exception as e:
        return f"News search failed: {str(e)}"

# List of all tools (LangChain will evaluate conditionally)
tools = [get_time, get_weather, search_news]

# ============= LLM SETUP =============
def get_llm(api_key: str, provider: str):
    if provider == "groq":
        return ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.7)
    else:  # openai
        return ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.7)

# Prompt for conditional tool triggering (case-based, not always) — FIXED with {tool_names}
REACT_PROMPT = PromptTemplate.from_template("""
You are a friendly SMS assistant. Answer naturally and concisely. Use current India time when relevant.

Available tools: {tool_names}
Tool descriptions: {tools}

Use tools ONLY when needed:
- get_time for time queries (e.g., 'What time is it?')
- get_weather for weather/location questions (e.g., 'Weather in Mumbai?')
- search_news for news/events (e.g., 'Latest news in India?')

Thought: Always reason if a tool is required. If not (e.g., general chat like 'Hello'), respond directly without tools.

Format: Thought: [reasoning] Action: [tool name] Action Input: [args] (only if tool needed)
Observation: [tool response]
... (repeat until done)
Final Answer: [response to user]

Question: {input}
{agent_scratchpad}
""")

# ============= AGENT EXECUTOR (Handles conditional tool calls) =============
def run_agent_with_tools(conversation_history: list, latest_message: str, api_key: str, provider: str = "groq"):
    llm = get_llm(api_key, provider)
    
    # Build full history for context
    messages = []
    for msg in conversation_history:
        role = "user" if msg.get("role") == "user" else "assistant"
        messages.append(HumanMessage(content=msg.get("content", "")) if role == "user" else SystemMessage(content=msg.get("content", "")))
    
    messages.append(HumanMessage(content=latest_message))
    
    # Create ReAct agent (conditional triggering via LLM reasoning)
    agent = create_react_agent(llm, tools, REACT_PROMPT)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=3)  # Limit loops for SMS brevity
    
    try:
        response = agent_executor.invoke({"input": latest_message, "chat_history": messages})
        return response["output"]  # Final agent response (tools called only if needed)
    except Exception as e:
        return f"Agent error: {str(e)} | Trace: {traceback.format_exc()[-200:]}"

# ============= FASTAPI APP =============
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
    provider = body.get("provider", "groq").lower()  # groq or openai
    
    if not message:
        return JSONResponse({"error": "Missing 'message'"}, status_code=400)
    if not api_key:
        return JSONResponse({"error": "Missing 'api_key' in request"}, status_code=400)
    if provider not in ["groq", "openai"]:
        return JSONResponse({"error": "Invalid provider. Use 'groq' or 'openai'"}, status_code=400)
    
    reply = run_agent_with_tools(conversation, message, api_key, provider)
    
    return JSONResponse({
        "reply": reply,
        "india_time_used": get_india_time(),  # Legacy fallback, but agent uses tool if needed
        "status": "success",
        "provider": provider,
        "tools_used": "LangChain ReAct (conditional)"
    })

@app.get("/")
def health():
    return {
        "status": "SMS AI Runtime Live — Tools enabled with conditional triggering",
        "provider": "groq/openai",
        "tools": ["get_time", "get_weather", "search_news"]
    }

# ============= CRITICAL: Auto-run when called via /int (exec) =============
if "inputs" in globals():
    try:
        data = globals().get("inputs", {})
        conversation = data.get("conversation", [])
        message = data.get("message", "")
        api_key = data.get("api_key", "")
        provider = data.get("provider", "groq").lower()
        
        if not message:
            result = {"error": "No message provided in payload"}
        elif not api_key:
            result = {"error": "No api_key provided in payload"}
        elif provider not in ["groq", "openai"]:
            result = {"error": "Invalid provider in payload"}
        else:
            reply = run_agent_with_tools(conversation, message, api_key, provider)
            result = {
                "reply": reply,
                "india_time_used": get_india_time(),
                "status": "success",
                "provider": provider,
                "tools_used": "LangChain ReAct (conditional)"
            }
        
        globals()["result"] = result
        
    except Exception as e:
        globals()["result"] = {
            "error": "Fatal execution error in main.py",
            "details": str(e),
            "traceback": traceback.format_exc()
        }
