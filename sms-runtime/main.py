# main.py - PURE LANGCHAIN DYNAMIC API TOOL (updated to fix PromptTemplate variable issues)
import os
import json
import requests
import traceback
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# LangChain
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# ============= DYNAMIC API LIST (you control this) =============
# Add as many as you want — just URL + description
DYNAMIC_APIS = [
    {
        "name": "get_india_time",
        "url": "https://worldtimeapi.org/api/timezone/Asia/Kolkata",
        "description": "Get current time in India. Use when user asks for time in India or 'what time is it?'"
    },
    {
        "name": "get_delhi_weather",
        "url": "https://api.open-meteo.com/v1/forecast?latitude=28.66&longitude=77.23&current_weather=true",
        "description": "Get current weather in Delhi. Use only for weather queries about Delhi."
    },
    {
        "name": "get_mumbai_weather",
        "url": "https://api.open-meteo.com/v1/forecast?latitude=19.07&longitude=72.87&current_weather=true",
        "description": "Get current weather in Mumbai. Use only for weather queries about Mumbai."
    },
    {
        "name": "get_news",
        "url": "https://newsapi.org/v2/top-headlines?country=in&apiKey=your_key",
        "description": "Get latest news in India. Use for news or current events questions."
    }
]

# ============= UNIVERSAL TOOL — CALL ANY API =============
@tool
def call_api(api_name: str) -> str:
    """Call any registered API by name. Only use if user query matches the description."""
    for api in DYNAMIC_APIS:
        if api["name"] == api_name:
            try:
                resp = requests.get(api["url"], timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    return json.dumps(data, indent=2)[:1000]  # Truncate if huge
                return f"API error {resp.status_code}: {resp.text[:200]}"
            except Exception as e:
                return f"API call failed: {str(e)}"
    return f"Unknown API: {api_name}"

tools = [call_api]

# ============= PROMPT — TELLS LLM ABOUT ALL APIS (fixed escaping and variables) =============
api_descriptions = "\n".join([f"- {api['name']}: {api['description']}" for api in DYNAMIC_APIS])

# Use a normal non f-string template and ensure any literal braces in the example JSON are escaped
# by using double curly braces so PromptTemplate does not interpret them as variables.
REACT_PROMPT = PromptTemplate.from_template(
    """
You are a helpful SMS assistant.

Available tools: {tool_names}
{tools}

API options (call using call_api):
""" + api_descriptions + """

Rules:
- Only call an API if the user query clearly matches its description
- If no API is needed (general chat), just respond normally
- Never make up data

Thought: Always reason step by step
Action: call_api
Action Input: {{"api_name": "api_name_here"}}  # Example: {{"api_name": "get_india_time"}}
Observation: [result]
... (repeat until done)
Final Answer: [your reply]

Question: {input}
{agent_scratchpad}
"""
)

# ============= AGENT (Pure LangChain) =============
def run_agent(conversation_history: list, message: str, api_key: str, provider: str):
    # Prepare LLM
    llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile", temperature=0.7) if provider == "groq" \
          else ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.7)

    # Create agent using the prompt and tools
    agent = create_react_agent(llm, tools, REACT_PROMPT)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=3,
        verbose=False
    )

    # Safely format conversation history into a plain text block so the prompt's {input} receives it
    try:
        if conversation_history and isinstance(conversation_history, list):
            parts = []
            for item in conversation_history:
                role = item.get("role", "user")
                content = item.get("content", "")
                parts.append(f"{role}: {content}")
            conv_text = "\n".join(parts)
        else:
            conv_text = ""

        combined_input = f"Conversation history:\n{conv_text}\n\nUser question: {message}"

        # Invoke the agent. Different langchain versions may return strings or dict-like results
        response = executor.invoke({
            "input": combined_input
        })

        # Normalize output from possible response shapes
        try:
            if isinstance(response, dict):
                # Common keys used by different LangChain runtimes
                if "output" in response:
                    return response["output"]
                if "result" in response:
                    return response["result"]
                # fallback to string conversion
                return json.dumps(response, default=str)
            elif isinstance(response, str):
                return response
            else:
                return str(response)
        except Exception:
            return f"Agent returned unrecognized response type: {repr(response)}"

    except Exception as e:
        tb = traceback.format_exc()
        return f"Agent error: {str(e)}\nTraceback:\n{tb}"

# ============= FASTAPI + /run-agent =============
app = FastAPI()

@app.post("/run-agent")
async def run(request: Request):
    body = await request.json()
    conv = body.get("conversation", [])
    msg = body.get("message", "")
    key = body.get("api_key", "")
    prov = body.get("provider", "groq")
    if not msg or not key:
        return JSONResponse({"error": "missing data"}, status_code=400)
    reply = run_agent(conv, msg, key, prov)
    return JSONResponse({
        "reply": reply,
        "status": "success",
        "provider": prov
    })

# /int support
if "inputs" in globals():
    data = globals().get("inputs", {})
    conv = data.get("conversation", [])
    msg = data.get("message", "")
    key = data.get("api_key", "")
    prov = data.get("provider", "groq")
    
    if msg and key:
        reply = run_agent(conv, msg, key, prov)
        globals()["result"] = {"reply": reply, "status": "success"}
    else:
        globals()["result"] = {"error": "missing message or key"}
