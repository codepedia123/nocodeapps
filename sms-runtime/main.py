# main.py - PURE LANGCHAIN DYNAMIC API TOOL (robust invoke + response parsing)
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
    try:
        for api in DYNAMIC_APIS:
            if api["name"] == api_name:
                try:
                    resp = requests.get(api["url"], timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                        # return a reasonable slice but allow more headroom than before
                        return json.dumps(data, indent=2)[:4000]
                    return f"API error {resp.status_code}: {resp.text[:400]}"
                except Exception as e:
                    return f"API call failed: {str(e)}"
        return f"Unknown API: {api_name}"
    except Exception as top_e:
        return f"call_api encountered an unexpected error: {str(top_e)}"

tools = [call_api]

# ============= PROMPT — TELLS LLM ABOUT ALL APIS (fixed escaping and variables) =============
api_descriptions = "\n".join([f"- {api['name']}: {api['description']}" for api in DYNAMIC_APIS])

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

# ============= Executor constructor helper (version tolerant) =============
def make_executor(agent_obj, tools_list, max_iterations=6, max_execution_time=None, verbose=False):
    """
    Construct AgentExecutor with fallback behavior depending on installed LangChain.
    """
    kwargs = dict(
        agent=agent_obj,
        tools=tools_list,
        handle_parsing_errors=True,
        max_iterations=max_iterations,
        verbose=verbose
    )
    # only add max_execution_time when not None
    if max_execution_time is not None:
        kwargs["max_execution_time"] = max_execution_time
    try:
        return AgentExecutor(**kwargs)
    except TypeError:
        # older/newer langchain may not accept max_execution_time or different names
        # try without it
        fallback_kwargs = kwargs.copy()
        fallback_kwargs.pop("max_execution_time", None)
        try:
            return AgentExecutor(**fallback_kwargs)
        except Exception as e:
            # last resort: minimal constructor
            try:
                return AgentExecutor(agent_obj, tools_list)
            except Exception as final_exc:
                raise final_exc

# ============= Response extraction helper =============
def extract_response_text(resp):
    """
    Given the raw object or dict returned by executor.invoke/invoke-like call,
    attempt to extract the most meaningful text.
    Handles:
    - plain string
    - dict with keys like 'output', 'result', 'text', 'answer', 'final_answer'
    - object with attributes 'output' or 'return_values'
    - lists
    - nested structures
    """
    try:
        if resp is None:
            return None

        # If it's already a string, return it
        if isinstance(resp, str):
            return resp

        # If it's bytes
        if isinstance(resp, (bytes, bytearray)):
            try:
                return resp.decode("utf-8", errors="replace")
            except Exception:
                return str(resp)

        # If it's a dict, search known keys
        if isinstance(resp, dict):
            # common keys
            for key in ("output", "result", "text", "answer", "final_answer", "final_output_text", "output_text", "response"):
                if key in resp and resp[key]:
                    return extract_response_text(resp[key])
            # some LangChain versions store return values under 'return_values'
            if "return_values" in resp and resp["return_values"]:
                return extract_response_text(resp["return_values"])
            # fallback: stringify relevant parts
            try:
                return json.dumps(resp, default=str)[:4000]
            except Exception:
                return str(resp)

        # If it's a list, join elements
        if isinstance(resp, (list, tuple)):
            parts = []
            for el in resp:
                parts.append(extract_response_text(el) or "")
            return "\n".join([p for p in parts if p])

        # If it's an object with attributes
        # try common attributes
        for attr in ("output", "result", "text", "answer", "return_values"):
            if hasattr(resp, attr):
                try:
                    val = getattr(resp, attr)
                    return extract_response_text(val)
                except Exception:
                    continue

        # If object has a .to_dict or .dict method
        if hasattr(resp, "to_dict"):
            try:
                return extract_response_text(resp.to_dict())
            except Exception:
                pass
        if hasattr(resp, "dict"):
            try:
                return extract_response_text(resp.dict())
            except Exception:
                pass

        # Fallback to repr/string
        return str(resp)[:4000]
    except Exception as e:
        return f"Failed to extract response text: {str(e)}"

# ============= Helper: detect iteration/time-limit messages ============
def indicates_iteration_or_time_limit(text_or_error):
    """
    Return True if the text or error message indicates the agent stopped due to iteration/time limits.
    """
    if text_or_error is None:
        return False
    text = str(text_or_error).lower()
    checks = [
        "iteration limit",
        "time limit",
        "stopped due to",
        "agent stopped",
        "stopped because",
        "stopped after",
        "exceeded max",
        "iteration",
        "max_iterations",
        "max execution time",
    ]
    for token in checks:
        if token in text:
            return True
    return False

# ============= AGENT (Pure LangChain) =============
def run_agent(conversation_history: list, message: str, api_key: str, provider: str):
    """
    Runs the LangChain React agent with robust invoke and response handling.
    """
    # Build LLM exactly as before
    llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile", temperature=0.7) if provider == "groq" \
          else ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.7)

    # Create agent using provided prompt and tools
    agent = create_react_agent(llm, tools, REACT_PROMPT)

    # Prepare conversation history text
    try:
        if conversation_history and isinstance(conversation_history, list):
            parts = []
            for item in conversation_history:
                if isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", "")
                else:
                    role = "user"
                    content = str(item)
                # Basic sanitization
                if isinstance(content, str):
                    content = content.replace("\x00", " ")
                parts.append(f"{role}: {content}")
            conv_text = "\n".join(parts)
        else:
            conv_text = ""
    except Exception:
        conv_text = ""

    combined_input = f"Conversation history:\n{conv_text}\n\nUser question: {message}"

    # Create executor with safe caps
    initial_max_iter = 6
    initial_max_time = 20  # advisory
    executor = make_executor(agent, tools, max_iterations=initial_max_iter, max_execution_time=initial_max_time, verbose=False)

    # Try to invoke. Prefer invoke over run, and handle many return shapes.
    def invoke_executor(exec_obj, inp):
        # prefer 'invoke' if present
        try:
            if hasattr(exec_obj, "invoke"):
                resp = exec_obj.invoke({"input": inp})
                return resp, None
            # older versions might expose 'arun' or callables; try calling the executor directly
            if callable(exec_obj):
                try:
                    resp = exec_obj({"input": inp})
                    return resp, None
                except TypeError:
                    # maybe it expects a string
                    resp = exec_obj(inp)
                    return resp, None
            # Last fallback: try 'run' but only if present and not deprecated in this environment.
            if hasattr(exec_obj, "run"):
                try:
                    resp = exec_obj.run(inp)
                    return resp, None
                except Exception as run_err:
                    return None, run_err
            return None, RuntimeError("Executor has no invoke/run/call method")
        except Exception as top_e:
            return None, top_e

    # Execute first attempt
    resp_raw, resp_err = invoke_executor(executor, combined_input)
    resp_text = extract_response_text(resp_raw)

    # If it looks like iteration/time-limit stop, retry once with higher caps
    if indicates_iteration_or_time_limit(resp_text) or indicates_iteration_or_time_limit(resp_err):
        # log for visibility
        print("Detected iteration/time-limit stop on first attempt. Retrying once with higher caps.")
        try:
            retry_max_iter = 12
            retry_max_time = 60
            executor_retry = make_executor(agent, tools, max_iterations=retry_max_iter, max_execution_time=retry_max_time, verbose=False)
            resp_raw_r, resp_err_r = invoke_executor(executor_retry, combined_input)
            resp_text_r = extract_response_text(resp_raw_r)

            # If retry produced a usable reply, return it
            if not indicates_iteration_or_time_limit(resp_text_r) and not indicates_iteration_or_time_limit(resp_err_r):
                # If retry had an error but produced text, include both
                if resp_err_r:
                    return f"{resp_text_r}\n\nNote: retry returned text but also an error: {repr(resp_err_r)}"
                return resp_text_r or "Agent returned empty response on retry."
            else:
                # Both attempts indicated iteration/time-limit; return detailed diagnostics
                diag = {
                    "first_response_text": resp_text,
                    "first_error_repr": repr(resp_err),
                    "retry_response_text": resp_text_r,
                    "retry_error_repr": repr(resp_err_r)
                }
                return "Agent stopped due to iteration/time limit after retry. Diagnostics: " + json.dumps(diag, default=str)
        except Exception as final_exc:
            tb = traceback.format_exc()
            return f"Agent retry failed with exception: {str(final_exc)}\nTraceback:\n{tb}"

    # If first attempt had an error but not iteration/time-limit, return helpful diagnostics
    if resp_err:
        tb = traceback.format_exc()
        # try to include any textual output we could extract
        return f"Agent execution error: {repr(resp_err)}\nExtracted text (if any):\n{resp_text}\nTraceback:\n{tb}"

    # Normal successful response
    return resp_text or "Agent returned no output."

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

# /int support (preserve original behavior)
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
