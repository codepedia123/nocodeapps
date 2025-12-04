# main.py - Real-time SMS AI Runtime (called from app.py via /int OR run as server)
import os
import json
import requests
import traceback
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

# Load .env once at startup (safe for both /int exec and uvicorn)
load_dotenv()

# ============= CONFIG =============
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============= TOOL: Get India Time =============
def get_india_time():
    try:
        resp = requests.get("https://worldtimeapi.org/api/timezone/Asia/Kolkata", timeout=5)
        data = resp.json()
        datetime_str = data["datetime"]
        time_only = datetime_str.split("T")[1][:8]  # HH:MM:SS
        return f"Current time in India (Kolkata): {time_only}"
    except Exception as e:
        return f"Time fetch failed: {str(e)}"

# ============= LLM CALL WITH FULL RAW ERROR LOGGING =============
def ask_llm(conversation_history: list, latest_message: str):
    india_time = get_india_time()

    # Build messages
    messages = [
        {"role": "system", "content": "You are a friendly SMS assistant. Always respond naturally and concisely. Use the current India time when relevant."}
    ]
    
    for msg in conversation_history:
        role = "user" if msg.get("role") == "user" else "assistant"
        messages.append({"role": role, "content": msg.get("content", "")})
    
    messages.append({"role": "user", "content": latest_message})
    messages.append({"role": "system", "content": india_time})

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

        # === FULL RAW ERROR REPORTING FROM PROVIDER ===
        if resp.status_code != 200:
            raw_error = resp.text.strip()
            try:
                error_json = resp.json()
                error_msg = error_json.get("error", {}).get("message", raw_error)
                error_type = error_json.get("error", {}).get("type", "unknown")
                return f"LLM ERROR ({resp.status_code}): {error_type} — {error_msg}"
            except:
                return f"LLM HTTP ERROR ({resp.status_code}): {raw_error[:200]}"

        result = resp.json()

        # Handle missing/invalid structure
        if "choices" not in result or not result["choices"]:
            return f"LLM returned no choices. Raw: {json.dumps(result)[:300]}"

        if "message" not in result["choices"][0] or "content" not in result["choices"][0]["message"]:
            return f"LLM malformed response. Raw: {json.dumps(result)[:300]}"

        return result["choices"][0]["message"]["content"].strip()

    except requests.exceptions.Timeout:
        return "LLM request timed out (15s)"
    except requests.exceptions.ConnectionError:
        return "LLM connection failed (network/DNS issue)"
    except requests.exceptions.RequestException as e:
        return f"LLM request failed: {str(e)}"
    except Exception as e:
        return f"Unexpected LLM error: {str(e)} | Trace: {traceback.format_exc()[-200:]}"

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
        "status": "success" if "ERROR" not in reply and "failed" not in reply.lower() else "llm_error",
        "provider": LLM_PROVIDER
    })

@app.get("/")
def health():
    return {
        "status": "SMS AI Runtime Live",
        "provider": LLM_PROVIDER,
        "model": "llama-3.3-70b-versatile" if LLM_PROVIDER == "groq" else "gpt-4o-mini"
    }

# ============= CRITICAL: Auto-run when called via /int (exec) =============
if "inputs" in globals():
    try:
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
                "status": "success" if "ERROR" not in reply and "failed" not in reply.lower() else "llm_error",
                "provider": LLM_PROVIDER
            }
        
        globals()["result"] = result
        
    except Exception as e:
        globals()["result"] = {
            "error": "Fatal execution error in main.py",
            "details": str(e),
            "traceback": traceback.format_exc()
        }
