# main.py - LangChain dynamic API runtime with full step-by-step logging
import os
import json
import requests
import traceback
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# LangChain imports
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# ---------------------------
# Logger helper (collects sequence of events)
# ---------------------------
class Logger:
    def __init__(self):
        self._events: List[Dict[str, Any]] = []

    def _now(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def log(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        entry = {
            "ts": self._now(),
            "id": str(uuid.uuid4()),
            "type": event_type,
            "message": message,
            "data": data or {}
        }
        self._events.append(entry)

    def to_list(self) -> List[Dict[str, Any]]:
        return self._events.copy()

    def clear(self):
        self._events = []

# Module-level logger instance used by tools and agent
logger = Logger()

# ---------------------------
# Tools: direct GET wrappers that log requests & responses
# ---------------------------
@tool
def get_india_time() -> str:
    """Get current time in India (Asia/Kolkata). No arguments."""
    event_id = str(uuid.uuid4())
    logger.log("tool.call", "get_india_time called", {"event_id": event_id})
    url = "https://worldtimeapi.org/api/timezone/Asia/Kolkata"
    start = time.time()
    try:
        resp = requests.get(url, timeout=10)
        elapsed = time.time() - start
        logger.log("http.request", "GET worldtimeapi", {"url": url, "status_code": resp.status_code, "elapsed_s": elapsed})
        resp.raise_for_status()
        data = resp.json()
        snippet = json.dumps(data, indent=2)[:2000]
        logger.log("tool.return", "get_india_time success", {"event_id": event_id, "snippet": snippet})
        return snippet
    except Exception as e:
        tb = traceback.format_exc()
        logger.log("tool.error", "get_india_time failed", {"event_id": event_id, "error": str(e), "traceback": tb})
        return f"get_india_time failed: {str(e)}"

@tool
def genderize(name: str) -> str:
    """Guess gender for a given name using genderize.io. Example: genderize('ishita')"""
    event_id = str(uuid.uuid4())
    logger.log("tool.call", "genderize called", {"event_id": event_id, "name": name})
    url = "https://api.genderize.io/"
    start = time.time()
    try:
        resp = requests.get(url, params={"name": name}, timeout=8)
        elapsed = time.time() - start
        logger.log("http.request", "GET genderize", {"url": url, "params": {"name": name}, "status_code": resp.status_code, "elapsed_s": elapsed})
        resp.raise_for_status()
        data = resp.json()
        snippet = json.dumps(data, indent=2)[:2000]
        logger.log("tool.return", "genderize success", {"event_id": event_id, "snippet": snippet})
        return snippet
    except Exception as e:
        tb = traceback.format_exc()
        logger.log("tool.error", "genderize failed", {"event_id": event_id, "name": name, "error": str(e), "traceback": tb})
        return f"genderize failed: {str(e)}"

@tool
def agify(name: str) -> str:
    """Guess age for a given name using agify.io. Example: agify('meelad')"""
    event_id = str(uuid.uuid4())
    logger.log("tool.call", "agify called", {"event_id": event_id, "name": name})
    url = "https://api.agify.io/"
    start = time.time()
    try:
        resp = requests.get(url, params={"name": name}, timeout=8)
        elapsed = time.time() - start
        logger.log("http.request", "GET agify", {"url": url, "params": {"name": name}, "status_code": resp.status_code, "elapsed_s": elapsed})
        resp.raise_for_status()
        data = resp.json()
        snippet = json.dumps(data, indent=2)[:2000]
        logger.log("tool.return", "agify success", {"event_id": event_id, "snippet": snippet})
        return snippet
    except Exception as e:
        tb = traceback.format_exc()
        logger.log("tool.error", "agify failed", {"event_id": event_id, "name": name, "error": str(e), "traceback": tb})
        return f"agify failed: {str(e)}"

@tool
def random_joke() -> str:
    """Get a random joke from official-joke-api.appspot.com. No arguments."""
    event_id = str(uuid.uuid4())
    logger.log("tool.call", "random_joke called", {"event_id": event_id})
    url = "https://official-joke-api.appspot.com/random_joke"
    start = time.time()
    try:
        resp = requests.get(url, timeout=8)
        elapsed = time.time() - start
        logger.log("http.request", "GET random_joke", {"url": url, "status_code": resp.status_code, "elapsed_s": elapsed})
        resp.raise_for_status()
        data = resp.json()
        snippet = json.dumps(data, indent=2)[:1500]
        logger.log("tool.return", "random_joke success", {"event_id": event_id, "snippet": snippet})
        return snippet
    except Exception as e:
        tb = traceback.format_exc()
        logger.log("tool.error", "random_joke failed", {"event_id": event_id, "error": str(e), "traceback": tb})
        return f"random_joke failed: {str(e)}"

# Register tools in a list used by the agent
tools = [get_india_time, genderize, agify, random_joke]

# ---------------------------
# Prompt: only safe variables {tool_names},{tools},{input},{agent_scratchpad}
# ---------------------------
api_descriptions = "\n".join([
    "- get_india_time(): Get current time in India (Asia/Kolkata).",
    "- genderize(name): Provide the person's name as single token, e.g. ishita.",
    "- agify(name): Provide the person's name as single token, e.g. meelad.",
    "- random_joke(): Returns a random joke (setup + punchline)."
])

REACT_PROMPT = PromptTemplate.from_template(
    """
You are a helpful SMS assistant that may call external tools when necessary.

Available tools: {tool_names}
{tools}

API options:
"""
    + api_descriptions
    + """

Usage examples (agent actions must use these tools only when required):
Action: get_india_time
Action Input: 

Action: genderize
Action Input: ishita

Action: agify
Action Input: meelad

Action: random_joke
Action Input: 

Rules:
- Call a tool only when the user's request explicitly requires it.
- When you call a tool, include only the required input token (name for genderize/agify) and return the tool output.
- Do not invent data or fabricate API outputs.
- Keep replies concise and accurate.

Question: {input}
{agent_scratchpad}
"""
)

# ---------------------------
# Executor constructor helper (version tolerant)
# ---------------------------
def make_executor(agent_obj, tools_list, max_iterations=6, max_execution_time=None, verbose=False):
    kwargs = dict(agent=agent_obj, tools=tools_list, handle_parsing_errors=True, max_iterations=max_iterations, verbose=verbose)
    if max_execution_time is not None:
        kwargs["max_execution_time"] = max_execution_time
    try:
        exec_inst = AgentExecutor(**kwargs)
        logger.log("executor.create", "AgentExecutor created", {"kwargs": {"max_iterations": max_iterations, "max_execution_time": max_execution_time}})
        return exec_inst
    except TypeError:
        # fallback
        try:
            kwargs.pop("max_execution_time", None)
            exec_inst = AgentExecutor(**kwargs)
            logger.log("executor.create", "AgentExecutor created (fallback)", {"kwargs": {"max_iterations": max_iterations}})
            return exec_inst
        except Exception as e:
            tb = traceback.format_exc()
            logger.log("executor.error", "AgentExecutor creation failed", {"error": str(e), "traceback": tb})
            raise

# ---------------------------
# Response extraction helper
# ---------------------------
def extract_response_text(resp) -> Optional[str]:
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
            # Common keys used across LangChain variants
            for key in ("output", "result", "text", "answer", "final_answer", "response"):
                if key in resp and resp[key]:
                    return extract_response_text(resp[key])
            if "return_values" in resp and resp["return_values"]:
                return extract_response_text(resp["return_values"])
            # include 'intermediate_steps' if present for diagnostics
            if "intermediate_steps" in resp:
                try:
                    return json.dumps(resp["intermediate_steps"], default=str)[:4000]
                except Exception:
                    pass
            return json.dumps(resp, default=str)[:4000]
        if isinstance(resp, (list, tuple)):
            parts = [extract_response_text(el) or "" for el in resp]
            return "\n".join([p for p in parts if p])
        # objects with common attributes
        for attr in ("output", "result", "text", "answer", "return_values"):
            if hasattr(resp, attr):
                try:
                    val = getattr(resp, attr)
                    return extract_response_text(val)
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
        return f"extract_response_text_error: {str(e)}"

# ---------------------------
# Detection helper (iteration/time-limit)
# ---------------------------
def indicates_iteration_or_time_limit(text_or_error) -> bool:
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
        "max execution time",
        "iteration"
    ]
    return any(tok in txt for tok in tokens)

# ---------------------------
# Main agent runner (returns dict with reply + logs + diagnostics)
# ---------------------------
def run_agent(conversation_history: List[Dict[str, Any]], message: str, provider: str, api_key_placeholder_shown: bool=False) -> Dict[str, Any]:
    """
    Run the agent. Returns a dict with:
      reply: final reply text (or diagnostic)
      logs: list of log events
      diagnostics: optional additional info
    Notes: This function does NOT log the api_key. Pass provider string only.
    """
    # clear logger for this request
    logger.clear()
    logger.log("run.start", "run_agent start", {"provider": provider})

    # Build LLM (do not log api_key)
    try:
        if provider == "groq":
            llm = ChatGroq(api_key=None, model="llama-3.3-70b-versatile", temperature=0.7)  # placeholder None to avoid logging
            logger.log("llm.create", "ChatGroq instance created (api_key omitted from logs)", {"model": "llama-3.3-70b-versatile"})
        else:
            llm = ChatOpenAI(api_key=None, model="gpt-4o-mini", temperature=0.7)
            logger.log("llm.create", "ChatOpenAI instance created (api_key omitted from logs)", {"model": "gpt-4o-mini"})
    except Exception as e:
        tb = traceback.format_exc()
        logger.log("llm.error", "LLM creation failed", {"error": str(e), "traceback": tb})
        return {"reply": f"LLM creation error: {str(e)}", "logs": logger.to_list(), "diagnostics": {"llm_error": str(e)}}

    # Create agent
    try:
        agent = create_react_agent(llm, tools, REACT_PROMPT)
        logger.log("agent.create", "Agent created", {})
    except Exception as e:
        tb = traceback.format_exc()
        logger.log("agent.error", "create_react_agent failed", {"error": str(e), "traceback": tb})
        return {"reply": f"Agent creation failed: {str(e)}", "logs": logger.to_list(), "diagnostics": {"agent_error": str(e)}}

    # Build conversation context text
    try:
        parts = []
        if conversation_history and isinstance(conversation_history, list):
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
    except Exception as e:
        conv_text = ""
        logger.log("conv.error", "Failed to build conversation text", {"error": str(e)})

    combined_input = f"Conversation history:\n{conv_text}\n\nUser question: {message}"
    logger.log("input.prepared", "Combined input prepared", {"combined_input_preview": combined_input[:2000]})

    # Executor creation & invocation
    initial_max_iter = 6
    initial_max_time = 20
    try:
        executor = make_executor(agent, tools, max_iterations=initial_max_iter, max_execution_time=initial_max_time, verbose=False)
    except Exception as e:
        tb = traceback.format_exc()
        logger.log("executor.error", "Failed to create executor", {"error": str(e), "traceback": tb})
        return {"reply": f"Executor creation failed: {str(e)}", "logs": logger.to_list(), "diagnostics": {"executor_error": str(e)}}

    def invoke_executor(exec_obj, inp) -> Tuple[Any, Optional[Exception]]:
        logger.log("executor.invoke.start", "Invoking executor", {"method_attempt_order": ["invoke","callable","run"]})
        try:
            if hasattr(exec_obj, "invoke"):
                resp = exec_obj.invoke({"input": inp})
                logger.log("executor.invoke", "Called invoke()", {"resp_type": type(resp).__name__})
                return resp, None
            if callable(exec_obj):
                try:
                    resp = exec_obj({"input": inp})
                    logger.log("executor.callable", "Called executor as callable with dict input", {"resp_type": type(resp).__name__})
                    return resp, None
                except TypeError:
                    resp = exec_obj(inp)
                    logger.log("executor.callable", "Called executor as callable with string input", {"resp_type": type(resp).__name__})
                    return resp, None
            if hasattr(exec_obj, "run"):
                resp = exec_obj.run(inp)
                logger.log("executor.run", "Called run()", {"resp_type": type(resp).__name__})
                return resp, None
            return None, RuntimeError("Executor has no invoke/run/call method")
        except Exception as e:
            tb = traceback.format_exc()
            logger.log("executor.invoke.error", "Executor invocation failed", {"error": str(e), "traceback": tb})
            return None, e

    # First attempt
    t0 = time.time()
    resp_raw, resp_err = invoke_executor(executor, combined_input)
    t1 = time.time()
    logger.log("executor.invoke.done", "Executor finished first attempt", {"elapsed_s": t1 - t0, "raw_type": type(resp_raw).__name__ if resp_raw is not None else None, "error": repr(resp_err) if resp_err else None})

    resp_text = extract_response_text(resp_raw)
    logger.log("executor.output.extract", "Extracted text from executor raw output", {"text_preview": resp_text[:2000] if resp_text else None})

    # If intermediate steps exist in raw dict, log them
    try:
        if isinstance(resp_raw, dict) and "intermediate_steps" in resp_raw:
            logger.log("executor.intermediate_steps", "Captured intermediate_steps", {"intermediate_steps": resp_raw.get("intermediate_steps")})
    except Exception:
        pass

    # Detect iteration/time-limit stops and retry once with higher caps if detected
    if indicates_iteration_or_time_limit(resp_text) or indicates_iteration_or_time_limit(resp_err):
        logger.log("executor.limit_detected", "Iteration/time-limit suspected on first attempt", {"resp_text_preview": str(resp_text)[:1000], "error_repr": repr(resp_err)})
        # Retry with higher limits
        try:
            retry_max_iter = 12
            retry_max_time = 60
            logger.log("executor.retry.start", "Starting retry with higher limits", {"retry_max_iter": retry_max_iter, "retry_max_time": retry_max_time})
            executor_retry = make_executor(agent, tools, max_iterations=retry_max_iter, max_execution_time=retry_max_time, verbose=False)
            t2 = time.time()
            resp_raw_r, resp_err_r = invoke_executor(executor_retry, combined_input)
            t3 = time.time()
            logger.log("executor.retry.done", "Retry finished", {"elapsed_s": t3 - t2, "raw_type": type(resp_raw_r).__name__ if resp_raw_r is not None else None, "error": repr(resp_err_r) if resp_err_r else None})
            resp_text_r = extract_response_text(resp_raw_r)
            logger.log("executor.retry.output.extract", "Extracted text from retry", {"text_preview": resp_text_r[:2000] if resp_text_r else None})

            if not indicates_iteration_or_time_limit(resp_text_r) and not indicates_iteration_or_time_limit(resp_err_r):
                # Success on retry
                final_reply = resp_text_r or "Agent returned empty response on retry."
                logger.log("run.success", "Agent returned output on retry", {"reply_preview": final_reply[:2000]})
                return {"reply": final_reply, "logs": logger.to_list(), "diagnostics": {"attempts": 2}}
            else:
                # Both failed; collect diagnostics
                diag = {
                    "first_response_text": resp_text,
                    "first_error_repr": repr(resp_err),
                    "retry_response_text": resp_text_r,
                    "retry_error_repr": repr(resp_err_r)
                }
                logger.log("run.failure", "Agent stopped due to iteration/time limit after retry", {"diagnostics": diag})
                return {"reply": "Agent stopped due to iteration/time limit after retry. See diagnostics.", "logs": logger.to_list(), "diagnostics": diag}
        except Exception as e:
            tb = traceback.format_exc()
            logger.log("executor.retry.exception", "Retry raised exception", {"error": str(e), "traceback": tb})
            return {"reply": f"Agent retry failed with exception: {str(e)}", "logs": logger.to_list(), "diagnostics": {"retry_exception": str(e)}}

    # If first attempt had an error (but not iteration/time-limit), return diagnostics
    if resp_err:
        tb = traceback.format_exc()
        logger.log("executor.error.final", "Executor returned error", {"error": str(resp_err)})
        return {"reply": "Agent execution error. See logs for details.", "logs": logger.to_list(), "diagnostics": {"error_repr": repr(resp_err), "extracted_text": resp_text}}

    # Success on first attempt
    logger.log("run.success", "Agent returned output", {"reply_preview": resp_text[:2000] if resp_text else None})
    return {"reply": resp_text or "Agent returned no output.", "logs": logger.to_list(), "diagnostics": {"attempts": 1}}

# ---------------------------
# FastAPI endpoints
# ---------------------------
app = FastAPI()

@app.post("/run-agent")
async def run(request: Request):
    body = await request.json()
    conv = body.get("conversation", [])
    msg = body.get("message", "")
    key = body.get("api_key", "")  # we will not log api_key anywhere
    prov = body.get("provider", "groq")
    if not msg or not key:
        return JSONResponse({"error": "missing data"}, status_code=400)

    # IMPORTANT: do not include api_key in logs or responses. We'll pass provider only.
    result = run_agent(conv, msg, provider=prov)
    # attach provider and status
    response = {
        "reply": result.get("reply"),
        "status": "success",
        "provider": prov,
        "logs": result.get("logs"),
        "diagnostics": result.get("diagnostics", {})
    }
    return JSONResponse(response)

# /int support for your execution environment (preserve behavior, set globals()["result"])
if "inputs" in globals():
    data = globals().get("inputs", {})
    conv = data.get("conversation", [])
    msg = data.get("message", "")
    key = data.get("api_key", "")
    prov = data.get("provider", "groq")

    if msg and key:
        _result = run_agent(conv, msg, provider=prov)
        globals()["result"] = {
            "reply": _result.get("reply"),
            "status": "success",
            "provider": prov,
            "logs": _result.get("logs"),
            "diagnostics": _result.get("diagnostics", {})
        }
    else:
        globals()["result"] = {"error": "missing message or key"}

# End of file
