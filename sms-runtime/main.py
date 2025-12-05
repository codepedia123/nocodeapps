# main.py - PURE LANGCHAIN DYNAMIC API TOOL (updated with four direct GET tools:
# get_india_time, genderize, agify, random_joke) and usage instructions in the prompt.
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

# ============= TOOLS (direct GET requests as requested) =============
# Each tool performs a single direct GET request to the endpoint the user specified.
# They return a short JSON/string slice so the agent can use them.
# - get_india_time() : no args, returns the full worldtimeapi response for Asia/Kolkata
# - genderize(name) : takes the name string the user asks about and returns genderize.io result
# - agify(name)    : takes the name string the user asks about and returns agify.io result
# - random_joke()  : no args, returns a random joke from official-joke-api.appspot.com

@tool
def get_india_time() -> str:
    """Get current time in India (Asia/Kolkata). No arguments."""
    try:
        resp = requests.get("https://worldtimeapi.org/api/timezone/Asia/Kolkata", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Return the full JSON as a string (bounded length)
        return json.dumps(data, indent=2)[:4000]
    except Exception as e:
        return f"get_india_time failed: {str(e)}"

@tool
def genderize(name: str) -> str:
    """Guess gender for a given name using genderize.io. Example usage: genderize(name='ishita')"""
    try:
        if not name:
            return "genderize: missing 'name' parameter"
        resp = requests.get("https://api.genderize.io/", params={"name": name}, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        return json.dumps(data, indent=2)[:4000]
    except Exception as e:
        return f"genderize failed: {str(e)}"

@tool
def agify(name: str) -> str:
    """Guess age for a given name using agify.io. Example usage: agify(name='meelad')"""
    try:
        if not name:
            return "agify: missing 'name' parameter"
        resp = requests.get("https://api.agify.io/", params={"name": name}, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        return json.dumps(data, indent=2)[:4000]
    except Exception as e:
        return f"agify failed: {str(e)}"

@tool
def random_joke() -> str:
    """Get a random joke from official-joke-api.appspot.com. No arguments."""
    try:
        resp = requests.get("https://official-joke-api.appspot.com/random_joke", timeout=8)
        resp.raise_for_status()
        data = resp.json()
        return json.dumps(data, indent=2)[:2000]
    except Exception as e:
        return f"random_joke failed: {str(e)}"

# Register tools in a list used by the agent
tools = [get_india_time, genderize, agify, random_joke]

# ============= PROMPT — DESCRIBES AVAILABLE TOOLS + USAGE INSTRUCTIONS ============
# Build usage descriptions programmatically so the prompt stays in sync with the tools.
api_descriptions = "\n".join([
    "- get_india_time(): Get current time in India (Asia/Kolkata). Use when user asks 'what time is it in India' or 'current time Kolkata'.",
    "- genderize(name): Provide the user's name as the 'name' parameter. Example: {\"name\": \"ishita\"}. Returns gender guess from genderize.io.",
    "- agify(name): Provide the user's name as the 'name' parameter. Example: {\"name\": \"meelad\"}. Returns age guess from agify.io.",
    "- random_joke(): Returns a random joke (setup + punchline). Use for light entertainment requests."
])

# Note: any literal JSON examples that include braces must be escaped using double braces {{ }} so PromptTemplate doesn't treat them as variables.
REACT_PROMPT = PromptTemplate.from_template(
    """
You are a helpful SMS assistant with access to four external helper tools. Use them only when they clearly match the user's request.

Available tools: {tool_names}
{tools}

API options and usage examples:
""" + api_descriptions + """

Usage instructions and examples:
- To get India time, call get_india_time() with no arguments.
- To guess gender for a name, call genderize with a name argument. Example action:
  Action: genderize
  Action Input: {{"name": "ishita"}}
- To guess age for a name, call agify with a name argument. Example action:
  Action: agify
  Action Input: {{"name": "meelad"}}
- To get a joke, call random_joke() with no arguments.

Rules:
- Only call a tool when the user's request explicitly requires it.
- Do not invent or hallucinate API outputs. Always return the tool output when you call a tool.
- Keep responses concise and accurate for the user's query.

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
    if max_execution_time is not None:
        kwargs["max_execution_time"] = max_execution_time
    try:
        return AgentExecutor(**kwargs)
    except TypeError:
        fallback_kwargs = kwargs.copy()
        fallback_kwargs.pop("max_execution_time", None)
        try:
            return AgentExecutor(**fallback_kwargs)
        except Exception:
            try:
                return AgentExecutor(agent_obj, tools_list)
            except Exception as final_exc:
                raise final_exc

# ============= Response extraction helper ============
def extract_response_text(resp):
    """
    Extract readable text from various possible executor responses.
    """
    try:
        if resp is None:
            return None
        if isinstance(resp, str):
            return resp
        if isinstance(resp, (bytes, bytearray)):
            try:
                return resp.decode("utf-8", errors="replace")
            except Exception:
                return str(resp)
        if isinstance(resp, dict):
            for key in ("output", "result", "text", "answer", "final_answer", "response"):
                if key in resp and resp[key]:
                    return extract_response_text(resp[key])
            if "return_values" in resp and resp["return_values"]:
                return extract_response_text(resp["return_values"])
            try:
                return json.dumps(resp, default=str)[:4000]
            except Exception:
                return str(resp)
        if isinstance(resp, (list, tuple)):
            parts = [extract_response_text(el) or "" for el in resp]
            return "\n".join([p for p in parts if p])
        # objects with attributes
        for attr in ("output", "result", "text", "answer", "return_values"):
            if hasattr(resp, attr):
                try:
                    return extract_response_text(getattr(resp, attr))
                except Exception:
                    continue
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
        return str(resp)[:4000]
    except Exception as e:
        return f"Failed to extract response text: {str(e)}"

def indicates_iteration_or_time_limit(text_or_error):
    """
    Detect phrases that indicate the agent hit iteration or time limits.
    """
    if text_or_error is None:
        return False
    txt = str(text_or_error).lower()
    tokens = [
        "iteration limit",
        "time limit",
        "stopped due to",
        "agent stopped",
        "exceeded max",
        "max_iterations",
        "max execution time"
    ]
    return any(tok in txt for tok in tokens)

# ============= AGENT (Pure LangChain) =============
def run_agent(conversation_history: list, message: str, api_key: str, provider: str):
    """
    Runs the agent using invoke() where possible and robustly extracts a textual reply.
    """
    # Build LLM (preserve provider logic)
    llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile", temperature=0.7) if provider == "groq" \
          else ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.7)

    agent = create_react_agent(llm, tools, REACT_PROMPT)

    # Prepare conversation context text
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
    initial_max_time = 20
    executor = make_executor(agent, tools, max_iterations=initial_max_iter, max_execution_time=initial_max_time, verbose=False)

    def invoke_executor(exec_obj, inp):
        try:
            if hasattr(exec_obj, "invoke"):
                resp = exec_obj.invoke({"input": inp})
                return resp, None
            if callable(exec_obj):
                try:
                    resp = exec_obj({"input": inp})
                    return resp, None
                except TypeError:
                    resp = exec_obj(inp)
                    return resp, None
            if hasattr(exec_obj, "run"):
                try:
                    resp = exec_obj.run(inp)
                    return resp, None
                except Exception as run_err:
                    return None, run_err
            return None, RuntimeError("Executor has no invoke/run/call method")
        except Exception as e:
            return None, e

    resp_raw, resp_err = invoke_executor(executor, combined_input)
    resp_text = extract_response_text(resp_raw)

    # Retry once with higher limits if we detect iteration/time-limit failure
    if indicates_iteration_or_time_limit(resp_text) or indicates_iteration_or_time_limit(resp_err):
        print("Detected iteration/time-limit stop on first attempt. Retrying once with higher limits.")
        try:
            retry_max_iter = 12
            retry_max_time = 60
            executor_retry = make_executor(agent, tools, max_iterations=retry_max_iter, max_execution_time=retry_max_time, verbose=False)
            resp_raw_r, resp_err_r = invoke_executor(executor_retry, combined_input)
            resp_text_r = extract_response_text(resp_raw_r)
            if not indicates_iteration_or_time_limit(resp_text_r) and not indicates_iteration_or_time_limit(resp_err_r):
                if resp_err_r:
                    return f"{resp_text_r}\n\nNote: retry returned text but also an error: {repr(resp_err_r)}"
                return resp_text_r or "Agent returned empty response on retry."
            else:
                diag = {
                    "first_response_text": resp_text,
                    "first_error_repr": repr(resp_err),
                    "retry_response_text": resp_text_r,
                    "retry_error_repr": repr(resp_err_r)
                }
                return "Agent stopped due to iteration/time limit after retry. Diagnostics: " + json.dumps(diag, default=str)
        except Exception as e:
            tb = traceback.format_exc()
            return f"Agent retry failed with exception: {str(e)}\nTraceback:\n{tb}"

    if resp_err:
        tb = traceback.format_exc()
        return f"Agent execution error: {repr(resp_err)}\nExtracted text (if any):\n{resp_text}\nTraceback:\n{tb}"

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
