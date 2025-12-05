# main.py - Real-time SMS AI Runtime (FINAL VERSION - NO MORE LOOPS)
import os
import json
import requests
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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

    return _cached_time or "Current time in India: unavailable"

# ============= TOOL DEFINITIONS (OpenAI Function Calling) =============
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time in India. Use when user asks for time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city. Use only for weather queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "Search latest news. Use only for news queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "News topic"}
                },
                "required": ["query"]
            }
        }
    }
]

# ============= TOOL IMPLEMENTATIONS =============
def execute_tool(name: str, args: dict):
    if name == "get_time":
        return get_india_time()
    if name == "get_weather":
        return "Weather in Mumbai: 28°C, partly cloudy"
    if name == "search_news":
        return "Latest news: India wins cricket match!"
    return "Unknown tool"

# ============= LLM CALL (OpenAI/Groq Function Calling) =============
def run_agent(conversation_history: list, latest_message: str, api_key: str, provider: str):
    messages = [{"role": "system", "content": "You are a helpful SMS assistant. Be concise. Use tools only when needed."}]
    
    for msg in conversation_history:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    
    messages.append({"role": "user", "content": latest_message})

    url = "https://api.groq.com/openai/v1/chat/completions" if provider == "groq" else "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile" if provider == "groq" else "gpt-4o-mini",
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.7,
        "max_tokens": 150
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        data = resp.json()

        if resp.status_code != 200:
            return f"LLM ERROR: {data.get('error', {}).get('message', 'Unknown')}"

        message = data["choices"][0]["message"]

        # If tool call
        if message.get("tool_calls"):
            tool_call = message["tool_calls"][0]
            name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"] or "{}")
            tool_result = execute_tool(name, args)
            
            # Second call with result
            messages.append(message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": name,
                "content": tool_result
            })
            
            second_resp = requests.post(url, json={**payload, "messages": messages}, headers=headers, timeout=15)
            second_data = second_resp.json()
            return second_data["choices"][0]["message"]["content"].strip()

        # Direct reply
        return message["content"].strip()

    except Exception as e:
        return f"Agent failed: {str(e)}"

# ============= FASTAPI =============
app = FastAPI()

@app.post("/run-agent")
async def run_agent_endpoint(request: Request):
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

    reply = run_agent(conversation, message, api_key, provider)

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
            reply = run_agent(conversation, message, api_key, provider)
            result = {
                "reply": reply,
                "india_time_used": get_india_time(),
                "status": "success"
            }

        globals()["result"] = result
    except Exception as e:
        globals()["result"] = {"error": str(e)}
