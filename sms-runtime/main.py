# main.py - PURE LANGCHAIN DYNAMIC API TOOL (robustified to avoid iteration/time-limit stops)
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
                        return json.dumps(data, indent=2)[:2000]  # truncate but allow larger output
                    return f"API error {resp.status_code}: {resp.text[:400]}"
                except Exception as e:
                    return f"API call failed: {str(e)}"
        return f"Unknown API: {api_name}"
    except Exception as top_e:
        return f"call_api encountered an unexpected error: {str(top_e)}"

tools = [call_api]

# ============= PROMPT — TELLS LLM ABOUT ALL APIS (fixed escaping and variables) =============
api_descriptions = "\n".join([f"- {api['name']}: {api['description']}" for api in DYNAMIC_APIS])

# The template must include {input} and {agent_scratchpad}. Any literal braces used in examples must be escaped.
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

# ============= Helper: safe executor constructor (handles langchain version differences) =============
def make_executor(agent_obj, tools_list, max_iterations=6, max_execution_time=None, verbose=False):
    """
    Construct AgentExecutor with graceful fallback if a parameter is unsupported by the installed LangChain.
    Returns the executor instance.
    """
    try:
        # Try with max_execution_time if supported
        if max_execution_time is not None:
            return AgentExecutor(
                agent=agent_obj,
                tools=tools_list,
                handle_parsing_errors=True,
                max_iterations=max_iterations,
                max_execution_time=max_execution_time,
                verbose=verbose
            )
        else:
            return AgentExecutor(
                agent=agent_obj,
                tools=tools_list,
                handle_parsing_errors=True,
                max_iterations=max_iterations,
                verbose=verbose
            )
    except TypeError:
        # Fallback: maybe constructor expects different keyword names or lacks max_execution_time
        try:
            return AgentExecutor(
                agent=agent_obj,
                tools=tools_list,
                handle_parsing_errors=True,
                max_iterations=max_iterations,
                verbose=verbose
            )
        except Exception as e:
            # As a last resort, call with minimal args
            return AgentExecutor(agent=agent_obj, tools=tools_list)

# ============= AGENT (Pure LangChain) =============
def run_agent(conversation_history: list, message: str, api_key: str, provider: str):
    """
    Runs the LangChain React agent with the supplied conversation and message.
    Implements robust execution and retries if the agent stops due to iteration or time limits.
    """
    # Build LLM (keep original provider logic unchanged)
    llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile", temperature=0.7) if provider == "groq" \
          else ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.7)

    # Create agent using the prompt and tools (preserve original function call style)
    agent = create_react_agent(llm, tools, REACT_PROMPT)

    # Prepare conversation text safely
    try:
        if conversation_history and isinstance(conversation_history, list):
            parts = []
            for item in conversation_history:
                role = item.get("role", "user") if isinstance(item, dict) else "user"
                content = item.get("content", "") if isinstance(item, dict) else str(item)
                # sanitize basic control characters
                if isinstance(content, str):
                    content = content.replace("\x00", " ")
                parts.append(f"{role}: {content}")
            conv_text = "\n".join(parts)
        else:
            conv_text = ""
    except Exception:
        conv_text = ""

    combined_input = f"Conversation history:\n{conv_text}\n\nUser question: {message}"

    # Execution strategy:
    # 1) Try with a moderate iteration budget and a guarded execution-time budget if supported.
    # 2) If result indicates iteration/time limit stop, retry once with higher limits.
    initial_max_iter = 6
    initial_max_time = 20  # seconds, advisory; may be ignored by some langchain versions

    executor = make_executor(agent, tools, max_iterations=initial_max_iter, max_execution_time=initial_max_time, verbose=False)

    # Helper to actually run the executor and capture result or exception
    def _execute_with_executor(exec_obj, inp):
        try:
            # prefer run(...) which returns final string in many LangChain versions
            if hasattr(exec_obj, "run"):
                return exec_obj.run(inp), None
            # some runtimes expose invoke
            if hasattr(exec_obj, "invoke"):
                resp = exec_obj.invoke({"input": inp})
                # try to extract string
                if isinstance(resp, dict):
                    for key in ("output", "result"):
                        if key in resp:
                            return resp[key], None
                    # if nothing extracted, return json dump
                    return json.dumps(resp, default=str), None
                return str(resp), None
            # fallback: try calling as a callable
            return str(exec_obj(inp)), None
        except Exception as e:
            return None, e

    # First attempt
    response_text, error = _execute_with_executor(executor, combined_input)

    # If there was an exception, or the agent returned the known "stopped due to iteration/time limit" message, attempt a single retry
    def _indicates_iteration_stop(res, err):
        if err is not None:
            msg = str(err).lower()
            if "iteration" in msg or "time limit" in msg or "stopped" in msg:
                return True
        if isinstance(res, str):
            low = res.lower()
            if "iteration limit" in low or "time limit" in low or "agent stopped" in low or "stopped due to" in low:
                return True
        return False

    if _indicates_iteration_stop(response_text, error):
        # Log the event to stdout for debugging in the environment
        print("Agent appears to have stopped due to iteration/time limit. Attempting one retry with higher limits.")
        try:
            # escalate limits but keep them bounded to avoid runaway loops
            retry_max_iter = 12
            retry_max_time = 60
            executor_retry = make_executor(agent, tools, max_iterations=retry_max_iter, max_execution_time=retry_max_time, verbose=False)
            response_text_retry, error_retry = _execute_with_executor(executor_retry, combined_input)

            # If retry succeeded, return it; otherwise prefer original failure with trace
            if not _indicates_iteration_stop(response_text_retry, error_retry):
                if error_retry:
                    # return both text (if any) and error trace
                    return f"{response_text_retry}\n\nNote: retry completed but produced an error: {str(error_retry)}"
                return response_text_retry
            else:
                # both attempts failed; build informative message including tracebacks where available
                err_msg = ""
                if error:
                    err_msg += f"First attempt error: {repr(error)}\n"
                if error_retry:
                    err_msg += f"Retry attempt error: {repr(error_retry)}\n"
                # include the textual outputs if any
                out_first = response_text or ""
                out_retry = response_text_retry or ""
                return ("Agent stopped due to iteration/time limit after retry. "
                        "First output:\n" + out_first + "\n\nRetry output:\n" + out_retry + "\n\nErrors:\n" + err_msg)
        except Exception as final_exc:
            tb = traceback.format_exc()
            return f"Agent retry failed with exception: {str(final_exc)}\nTraceback:\n{tb}"

    # If original run produced an exception, return diagnostic
    if error is not None:
        tb = traceback.format_exc()
        return f"Agent execution error: {str(error)}\nTraceback:\n{tb}"

    # Normal successful return
    return response_text if response_text is not None else "Agent returned no output."

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
