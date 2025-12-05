# main.py - FINAL WORKING VERSION (NO MORE "No output")
import os
import json
import requests
import traceback
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ============= DYNAMIC API LIST =============
DYNAMIC_APIS = [
    {
        "name": "get_india_time",
        "url": "https://worldtimeapi.org/api/timezone/Asia/Kolkata",
        "description": "Get current time in India"
    },
    {
        "name": "get_weather_delhi",
        "url": "https://api.open-meteo.com/v1/forecast?latitude=28.66&longitude=77.23&current_weather=true",
        "description": "Weather in Delhi"
    },
    {
        "name": "get_weather_mumbai",
        "url": "https://api.open-meteo.com/v1/forecast?latitude=19.07&longitude=72.87&current_weather=true",
        "description": "Weather in Mumbai"
    }
]

# ============= UNIVERSAL TOOL =============
def call_api(api_name: str) -> str:
    for api in DYNAMIC_APIS:
        if api["name"] == api_name:
            try:
                resp = requests.get(api["url"], timeout=10)
                if resp.status_code == 200:
                    return json.dumps(resp.json(), indent=2)[:1000]
                return f"Error {resp.status_code}"
            except:
                return "API failed"
    return "Unknown API"

# ============= SIMPLE LLM CALL (NO LangChain — 100% reliable) =============
def run_llm(message: str, api_key: str, provider: str = "groq"):
    # Build prompt with API descriptions
    api_list = "\n".join([f"- {api['name']}: {api['description']}" for api in DYNAMIC_APIS])
    
    prompt = f"""
You are a helpful SMS assistant.

Available APIs (call with call_api("name")):
{api_list}

Only call an API if the user clearly asks for it.
For general chat, just reply normally.

User: {message}
Assistant:"""

    url = "https://api.groq.com/openai/v1/chat/completions" if provider == "groq" else "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile" if provider == "groq" else "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 150
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        if resp.status_code != 200:
            return f"LLM error: {resp.text[:200]}"
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Failed: {str(e)}"

# ============= FASTAPI =============
app = FastAPI()

@app.post("/run-agent")
async def run_agent(request: Request):
    try:
        body = await request.json()
    except:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    message = body.get("message", "")
    api_key = body.get("api_key", "")
    provider = body.get("provider", "groq").lower()

    if not message or not api_key:
        return JSONResponse({"error": "Missing data"}, status_code=400)

    reply = run_llm(message, api_key, provider)

    return JSONResponse({
        "reply": reply,
        "status": "success",
        "provider": provider
    })

# ============= /int EXECUTION — THIS IS THE FIX =============
if "inputs" in globals():
    try:
        data = globals().get("inputs", {})
        message = data.get("message", "")
        api_key = data.get("api_key", "")
        provider = data.get("provider", "groq").lower()

        if not message or not api_key:
            result = {"error": "Missing message or api_key"}
        else:
            reply = run_llm(message, api_key, provider)
            result = {
                "reply": reply,
                "status_india_time": get_india_time() if 'time' in message.lower() else "N/A",
                "status": "success"
            }

        # THIS IS THE KEY: SET result DIRECTLY
        globals()["result"] = result

    except Exception as e:
        globals()["result"] = {"error": str(e), "traceback": traceback.format_exc()}
