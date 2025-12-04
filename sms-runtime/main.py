# main.py - Real-time SMS AI Runtime (Twilio → this file)
import os
import json
import redis.cluster
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.tools import tool
import requests

load_dotenv()

# ================= REDIS CLUSTER (your Valkey) =================
def get_redis():
    url = os.getenv("REDIS_URL", "rediss://smsruntime-sm3cdo.serverless.use1.cache.amazonaws.com:6379")
    return redis.cluster.RedisCluster.from_url(
        url,
        decode_responses=True,
        ssl_cert_reqs=None,
        socket_connect_timeout=3,
        socket_timeout=3,
    )

r = get_redis()

# ================= CONFIG =================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_CONFIG = {
    "groq": {
        "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-instruct"),
        "api_key": os.getenv("GROQ_API_KEY"),
        "temperature": 0.3,
    },
    "openai": {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0.3,
    },
}

# ================= TOOLS =================
@tool
def trigger_workflow(workflow_id: str, data: dict):
    """Call Node-RED or your internal function"""
    url = f"https://yourdomain.com/webhook/{workflow_id}"  # change later
    try:
        resp = requests.post(url, json=data, timeout=20)
        return resp.json()
    except:
        return {"error": "workflow failed"}

tools = [trigger_workflow]

# ================= LLM =================
def get_llm():
    cfg = LLM_CONFIG[LLM_PROVIDER]
    if LLM_PROVIDER == "groq":
        return ChatGroq(**cfg)
    return ChatOpenAI(**cfg)

# ================= FASTAPI =================
app = FastAPI()

@app.post("/webhook/sms")
async def sms_webhook(request: Request):
    form = await request.form()
    phone = form.get("From")
    message = form.get("Body", "").strip()
    agent_id = form.get("agent_id")  # You will pass this from Twilio URL

    if not (phone and agent_id):
        return PlainTextResponse("Bad request", status_code=400)

    # Load agent prompt
    agent_data = r.hgetall(agent_id)
    if not agent_data:
        return PlainTextResponse("Agent not found", status_code=404)
    prompt = agent_data.get("prompt", "You are a helpful SMS assistant.")

    # Load history
    history_key = f"convo:{agent_id}:{phone}"
    raw = r.get(history_key)
    messages = json.loads(raw) if raw else []

    # Run agent
    agent = create_react_agent(get_llm(), tools)
    inputs = {
        "messages": [
            {"role": "system", "content": prompt},
            *messages,
            {"role": "user", "content": message}
        ]
    }
    response = ""
    for chunk in agent.stream(inputs):
        if "agent" in chunk and chunk["agent"]["messages"]:
            msg = chunk["agent"]["messages"][-1]
            if hasattr(msg, "content"):
                response += msg.content

    # Save history
    messages.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": response}
    ])
    r.set(history_key, json.dumps(messages[-50:]), ex=60*60*24*30)

    return PlainTextResponse(response.strip() or "I'm here!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
