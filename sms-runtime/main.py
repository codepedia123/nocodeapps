# main.py - PURE LANGCHAIN DYNAMIC API TOOL (fixed PromptTemplate variable issue)
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

# ============= TOOLS (direct GET requests) =============
@tool
def get_india_time() -> str:
    """Get current time in India (Asia/Kolkata). No arguments."""
    try:
        resp = requests.get("https://worldtimeapi.org/api/timezone/Asia/Kolkata", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return json.dumps(data, indent=2)[:4000]
    except Exception as e:
        return f"get_india_time failed: {str(e)}"

@tool
def genderize(name: str) -> str:
    """Guess gender for a given name using genderize.io. Example: genderize('ishita')"""
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
    """Guess age for a given name using agify.io. Example: agify('meelad')"""
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

tools = [get_india_time, genderize, agify, random_joke]

# ============= PROMPT — DESCRIBES AVAILABLE TOOLS and usage instructions ============
api_descriptions = "\n".join([
    "- get_india_time(): Get current time in India (Asia/Kolkata). Use for questions like 'what time is it in India' or 'Kolkata time'.",
    "- genderize(name): Provide the person's name as a single string. Example usage in agent actions: Action: genderize  Action Input: ishita",
    "- agify(name): Provide the person's name as a single string. Example usage in agent actions: Action: agify  Action Input: meelad",
    "- random_joke(): Returns a random joke with setup and punchline. Use for light entertainment requests."
])

# IMPORTANT: only these template variables appear: {tool_names}, {tools}, {input}, {agent_scratchpad}
# Do not include any other {curly} tokens to avoid PromptTemplate variable errors.
REACT_PROMPT = PromptTemplate.from_template(
    """
You are a helpful SMS assistant with four external helper tools. Use the tools only when the user's request clearly requires them.

Available tools: {tool_names}
{tools}

API options and concise usage instructions:
"""
    + api_descriptions
    + """

Usage examples for the agent:
To get India time, call get_india_time with no arguments. Example action:
Action: get_india_time
Action Input: 

To guess gender, call genderize with the name as a single token. Example action:
Action: genderize
Action Input: ishita

To guess age, call agify with the name as a single token. Example action:
Action: agify
Action Input: meelad

To get a joke, call random_joke with no arguments. Example action:
Action: random_joke
Action Input: 

Rules:
- Only call a tool when the user's request explicitly demands it.
- When you call a tool, return the tool output directly and then continue reasoning if needed.
- Do not fabricate or invent API outputs.
- Keep replies concise and accurate.

Question: {input}
{agent_scratchpad}
"""
)

# ============= Executor constructor helper (version tolerant) =============
def make_executor(agent_obj, tools_list, max_iterations=6, max_execution_time=None, verbose=False):
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
        # fallback without max_execution_time
        kwargs.pop("max_execution_time", None)
        try:
            return AgentExecutor(**kwargs)
        except Exception:
            try:
                return AgentExecutor(agent_obj, tools_list)
            except Exception as final_exc:
                raise final_exc

# ============= Response extraction helper ============
def extract_response_text(resp):
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
    llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile", temperature=0.7) if provider == "groq" \
          else ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.7)

    agent = create_react_agent(llm, tools, REACT_PROMPT)

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
