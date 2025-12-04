# main.py - Real-time SMS AI Runtime (called from app.py via /int OR run as server)
import os
import json
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

# Load .env once at startup (safe for both /int exec and uvicorn)
load_dotenv()

# ============= CONFIG =============
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = "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"

# ============= TOOL: Get India Time =============
def get_india_time():
    try:
        resp = requests.get("https://worldtimeapi.org/api/timezone/Asia/Kolkata", timeout=5)
        data = resp.json()
        datetime_str = data["datetime"]
        time_only = datetime_str.split("T")[1][:8]  # HH:MM:SS
        return f"Current time in India (Kolkata): {time_only}"
    except:
        return "I couldn't fetch the time right now."

# ============= LLM CALL (Groq or OpenAI) =============
def ask_llm(conversation_history: list, latest_message: str):
    india_time = get_india_time()
    # Build messages
    messages = [
        {"role": "system", "content": "You are a friendly SMS assistant. Always respond naturally and concisely. Use the current India time when relevant."}
    ]
    
    # Add past conversation
    for msg in conversation_history:
        role = "user" if msg.get("role") == "user" else "assistant"
        messages.append({"role": role, "content": msg.get("content", "")})
    
    # Add latest user message
    messages.append({"role": "user", "content": latest_message})
    
    # Add current time as system message
    messages.append({"role": "system", "content": india_time})
    
    # === CALL LLM ===
    url = "https://api.groq.com/openai/v1/chat/completions" if LLM_PROVIDER == "groq" else "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY if LLM_PROVIDER == 'groq' else OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile" if LLM_PROVIDER == "groq" else "gpt-4o-mini",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 150
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        result = resp.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Sorry, I'm having trouble replying right now. ({str(e)})"

# ============= FASTAPI APP =============
app = FastAPI()

@app.post("/run-agent")
async def run_agent(request: Request):
    try:
        body = await request.json()
    except:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    
    conversation = body.get("conversation", [])
    latest_message = body.get("message", "").strip()
    
    if not latest_message:
        return JSONResponse({"error": "Missing 'message'"}, status_code=400)
    
    reply = ask_llm(conversation, latest_message)
    
    return JSONResponse({
        "reply": reply,
        "india_time_used": get_india_time(),
        "status": "success"
    })

@app.get("/")
def health():
    return {"status": "SMS AI Runtime Live", "provider": LLM_PROVIDER}


# ============= CRITICAL: Auto-run when called via /int (exec) =============
# This block runs ONLY when executed via app.py's /int endpoint
if "inputs" in globals():
    try:
        # Extract data exactly like app.py's injection (inputs = combined_input = payload dict)
        data = globals().get("inputs", {})
        conversation = data.get("conversation", [])
        message = data.get("message", "")
        
        if not message:
            result = {"error": "No message provided in payload"}
        else:
            reply = ask_llm(conversation, message)
            result = {
                "reply": reply,
                "india_time_used": get_india_time(),
                "status": "success"
            }
        
        # This makes /int return the real result
        globals()["result"] = result
        
    except Exception as e:
        globals()["result"] = {"error": str(e), "traceback": traceback.format_exc()}
