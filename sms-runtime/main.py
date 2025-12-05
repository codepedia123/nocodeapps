# main.py - Real-time SMS AI Runtime (LangChain ReAct Agent + Conditional Tools)
import os
import json
import requests
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ============= BULLETPROOF INDIA TIME TOOL (Multiple Fallbacks + Cache) =============
import time
_cached_time = None
_cached_at = 0

def get_india_time():
    global _cached_time, _cached_at
    now = time.time()
    if _cached_time and (now - _cached_at) < 30:
        return _cached_time

    urls = [
        "https://worldtimeapi.org/api/timezone/Asia/Kolkata",
        "http://worldtimeapi.org/api/timezone/Asia/Kolkata",
        "https://timeapi.io/api/Time/current/zone?timeZone=Asia/Kolkata",
        "https://worldclockapi.com/api/json/ist/now"
    ]

    for url in urls:
        try:
            resp = requests.get(url, timeout=7)
            if resp.status_code != 200:
                continue
            data = resp.json()
            dt = data.get("datetime") or data.get("dateTime") or data.get("currentDateTime", "")
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
    return "Current time in India: [unavailable]"

# ============= LANGCHAIN TOOLS (Only Triggered When Needed) =============
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class TimeInput(BaseModel):
    timezone: str = Field(default="Asia/Kolkata", description="Timezone like 'Asia/Kolkata'")

@tool(args_schema=TimeInput)
def get_current_time(timezone: str = "Asia/Kolkata") -> str:
    """Get current time in any timezone. Use when user asks 'What time is it in [place]?' or similar."""
    try:
        resp = requests.get(f"https://worldtimeapi.org/api/timezone/{timezone}", timeout=7)
        if resp.status_code == 200:
            data = resp.json()
            dt = data["datetime"]
            time_str = dt.split("T")[1][:8]
            return f"Current time in {timezone}: {time_str}"
        return "Time unavailable for that zone"
    except:
        return get_india_time()  # fallback to India time

class WeatherInput(BaseModel):
    location: str = Field(..., description="City name, e.g., Mumbai, Delhi")

@tool(args_schema=WeatherInput)
def get_weather(location: str) -> str:
    """Get current weather. Trigger only when user asks about weather, temperature, or climate."""
    # Replace with your real API key or make it user-provided
    API_KEY = os.getenv("OPENWEATHER_KEY", "your_key_here")
    url = "https://api.openweathermap.org/data/2.5/weather"
    try:
        resp = requests.get(url, params={"q": location, "appid": API_KEY, "units": "metric"}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"].title()
            return f"Weather in {location}: {desc}, {temp}°C"
        return f"Could not fetch weather for {location}"
    except:
        return "Weather service unavailable"

# Add more tools as needed (news, calendar, etc.)

tools = [get_current_time, get_weather]

# ============= LANGCHAIN AGENT SETUP (ReAct + Conditional Tool Use) =============
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

def create_agent(api_key: str, provider: str = "groq"):
    if provider == "groq":
        llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile", temperature=0.7)
    else:
        llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.7)

    prompt = PromptTemplate.from_template("""
You are a friendly, natural SMS chatbot. Respond concisely.

You have access to tools. Use them ONLY when needed:
- get_current_time: When user asks about time in any location
- get_weather: When user asks about weather, temperature, or climate

Thought: Always think: "Do I need a tool? If not, just reply normally."

Example:
User: "Hey what's up?" → No tool → Just chat
User: "What's the time in London?" → Use get_current_time
User: "Is it raining in Mumbai?" → Use get_weather

Respond naturally. Never mention tools.

{tools}
{agent_scratchpad}
""")

    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

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
    provider = body.get("provider", "groq").lower()

    if not message:
        return JSONResponse({"error": "Missing 'message'"}, status_code=400)
    if not api_key:
        return JSONResponse({"error": "Missing 'api_key'"}, status_code=400)
    if provider not in ["groq", "openai"]:
        return JSONResponse({"error": "Invalid provider"}, status_code=400)

    # Convert conversation to LangChain format
    history = []
    for msg in conversation:
        role = "human" if msg.get("role") == "user" else "assistant"
        history.append({"role": role, "content": msg.get("content", "")})

    try:
        agent_executor = create_agent(api_key, provider)
        response = agent_executor.invoke({
            "input": message,
            "chat_history": history
        })
        reply = response["output"]
    except Exception as e:
        reply = f"Agent error: {str(e)}"

    return JSONResponse({
        "reply": reply,
        "india_time_used": get_india_time(),
        "status": "success",
        "provider": provider,
        "tools_used": "conditional (only when needed)"
    })

@app.get("/")
def health():
    return {"status": "LangChain SMS Agent Live", "mode": "case-based tool calling"}

# ============= /int EXEC SUPPORT =============
if "inputs" in globals():
    try:
        data = globals().get("inputs", {})
        conversation = data.get("conversation", [])
        message = data.get("message", "")
        api_key = data.get("api_key", "")
        provider = data.get("provider", "groq").lower()

        if not all([message, api_key]):
            result = {"error": "Missing message or api_key"}
        elif provider not in ["groq", "openai"]:
            result = {"error": "Invalid provider"}
        else:
            history = [{"role": "human" if m.get("role") == "user" else "assistant", "content": m.get("content", "")} for m in conversation]
            agent_executor = create_agent(api_key, provider)
            response = agent_executor.invoke({"input": message, "chat_history": history})
            reply = response["output"]
            result = {
                "reply": reply,
                "india_time_used": get_india_time(),
                "status": "success",
                "provider": provider
            }

        globals()["result"] = result

    except Exception as e:
        globals()["result"] = {"error": "Agent failed", "details": str(e)}
