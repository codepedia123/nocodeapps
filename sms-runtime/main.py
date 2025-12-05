# main.py - Real-time SMS AI Runtime (called from app.py via /int OR run as server)
import os
import json
import requests
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

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

# ============= LLM CALL WITH USER-PROVIDED API KEY + FULL RAW ERROR LOGGING =============
def ask_llm(conversation_history: list, latest_message: str, api_key: str, provider: str = "groq"):
    india_time = get_india_time()

    messages = [
        {"role": "system", "content": "You are a friendly SMS assistant. Always respond naturally and concisely. Use the current India time when relevant."}
    ]
    
    for msg in conversation_history:
        role = "user" if msg.get("role") == "user" else "assistant"
        messages.append({"role": role, "content": msg.get("content", "")})
    
    messages.append({"role": "user", "content": latest_message})
    messages.append({"role": "system", "content": india_time})

    if provider == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        model = "llama-3.3-70b-versatile"
    else:  # openai
        url = "https://api.openai.com/v1/chat/completions"
        model = "gpt-4o-mini"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 150
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)

        if resp.status_code != 200:
            raw_error = resp.text.strip()
            try:
                error_json = resp.json()
                err_msg = error_json.get("error", {}).get("message", raw_error)
                err_type = error_json.get("error", {}).get("type", "unknown")
                return f"LLM ERROR ({resp.status_code}): {err_type} — {err_msg}"
            except:
                return f"LLM HTTP ERROR ({resp.status_code}): {raw_error[:300]}"

        result = resp.json()
        if "choices" not in result or not result["choices"]:
            return f"LLM returned no choices. Raw: {json.dumps(result)[:300]}"
        if "message" not in result["choices"][0] or "content" not in result["choices"][0]["message"]:
            return f"LLM malformed response. Raw: {json.dumps(result)[:300]}"

        return result["choices"][0]["message"]["content"].strip()

    except requests.exceptions.Timeout:
        return "LLM request timed out (15s)"
    except requests.exceptions.ConnectionError:
        return "LLM connection failed"
    except requests.exceptions.RequestException as e:
        return f"LLM request failed: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)} | Trace: {traceback.format_exc()[-200:]}"

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

    reply = ask_llm(conversation, message, api_key, provider)

    return JSONResponse({
        "reply": reply,
        "india_time_used": get_india_time(),
        "status": "success" if "ERROR" not in reply.upper() and "failed" not in reply.lower() else "llm_error",
        "provider": provider
    })

@app.get("/")
def health():
    return {"status": "SMS AI Runtime Live — Send api_key in request"}

# ============= CRITICAL: Auto-run when called via /int (exec) =============
# Supports multiple injection styles:
# 1) Direct globals keys (message, api_key, provider, conversation)
# 2) payload dict
# 3) input dict
# 4) legacy inputs dict
if any(k in globals() for k in ["inputs", "payload", "input", "message", "api_key", "provider", "conversation"]):
    try:
        g = globals()

        payload = g.get("payload", {})
        input_data = g.get("input", {})
        inputs_dict = g.get("inputs", {})

        combined_input = {}

        # priority: payload > input_data > raw globals (including legacy inputs dict)
        if isinstance(payload, dict):
            combined_input.update(payload)
        if isinstance(input_data, dict):
            combined_input.update(input_data)
        if isinstance(inputs_dict, dict):
            for k, v in inputs_dict.items():
                if k not in combined_input:
                    combined_input[k] = v

        # final fallback: direct injected keys
        for key in ["conversation", "message", "api_key", "provider"]:
            if key in g and key not in combined_input:
                combined_input[key] = g.get(key)

        conversation = combined_input.get("conversation", [])
        message = combined_input.get("message", "")
        api_key = combined_input.get("api_key", "")
        provider = combined_input.get("provider", "groq")

        # Defensive normalization
        if isinstance(conversation, str):
            try:
                conversation = json.loads(conversation)
            except:
                conversation = []

        if not isinstance(conversation, list):
            conversation = []

        message = (message or "")
        if not isinstance(message, str):
            message = str(message)

        api_key = (api_key or "")
        if not isinstance(api_key, str):
            api_key = str(api_key)

        provider = (provider or "groq")
        if not isinstance(provider, str):
            provider = str(provider)
        provider = provider.lower()

        message = message.strip()
        api_key = api_key.strip()

        if not message:
            result = {"error": "No message provided in payload"}
        elif not api_key:
            result = {"error": "No api_key provided in payload"}
        elif provider not in ["groq", "openai"]:
            result = {"error": "Invalid provider in payload"}
        else:
            reply = ask_llm(conversation, message, api_key, provider)
            result = {
                "reply": reply,
                "india_time_used": get_india_time(),
                "status": "success" if "ERROR" not in reply.upper() and "failed" not in reply.lower() else "llm_error",
                "provider": provider
            }
        
        g["result"] = result

    except Exception as e:
        globals()["result"] = {
            "error": "Fatal execution error in main.py",
            "details": str(e),
            "traceback": traceback.format_exc()
        }
