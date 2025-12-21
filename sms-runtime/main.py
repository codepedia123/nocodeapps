# main.py - Modern Tool Calling Agent (Dynamic Decision Making)
import os
import json
import requests
import traceback
import time
import uuid
import urllib.parse
import socket
import subprocess
import shlex
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# LangChain modern imports
from langchain_core.tools import StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# ---------------------------
# Configuration JSON
# ---------------------------
# Keep a minimal default fallback config. This will be merged with fetched tools at runtime.
DYNAMIC_CONFIG = {
    "5": {
        "api_url": "https://ap.rhythmflow.ai/api/v1/webhooks/UZ6KJw8w1EInlLqhP1gWZ/sync",
        "api_payload_json": "%7B%22values%22%3A%7B%22A%22%3A%22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C%22B%22%3A%22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C%22C%22%3A%22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%7D%7D",
        "instructions": "'values.A' is Name; 'values.B' is Email; 'values.C' is Phone.",
        "when_run": "ONLY run this when the user specifically provides an email or contact info to be saved/synced."
    }
}

# ---------------------------
# Logger helper
# ---------------------------
class Logger:
    def __init__(self):
        self._events: List[Dict[str, Any]] = []
    def _now(self) -> str:
        return datetime.utcnow().isoformat() + "Z"
    def log(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        entry = {"ts": self._now(), "id": str(uuid.uuid4()), "type": event_type, "message": message, "data": data or {}}
        self._events.append(entry)
    def to_list(self) -> List[Dict[str, Any]]: return self._events.copy()
    def clear(self): self._events = []

logger = Logger()

# ---------------------------
# Helper: fetch agent details and tools from remote DB endpoints
# ---------------------------
# NOTE: adapted so tools fetching uses the same IP-based API first, then falls back to domain.
FETCH_BASE = "http://54.89.185.235:8000/fetch"
DOMAIN_FALLBACK = "https://api.rhythmflow.ai/fetch"

def fetch_agent_details(agent_id: str, timeout: int = 30):
    """
    Robust fetch:
    1) Try IP-based endpoint first (no TLS, avoids DNS/proxy/TLS surprises).
    2) If that fails, try domain endpoint with TLS verification.
    3) If that fails, try domain endpoint with verify=False for diagnostic purposes.
    4) If still failing, run TCP checks and a subprocess curl for diagnostics and log them.
    Returns parsed JSON dict on success, otherwise None.
    """
    ip_url = f"http://54.89.185.235:8000/fetch"
    domain_url = "https://api.rhythmflow.ai/fetch"
    params = {"table": "agents", "id": agent_id}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }

    session = requests.Session()
    session.trust_env = True

    def try_get(url, params, verify=True, timeout_local=timeout):
        try:
            logger.log("fetch.attempt", "HTTP GET", {"url": url, "params": params, "verify": verify})
            resp = session.get(url, params=params, headers=headers, timeout=timeout_local, allow_redirects=True, verify=verify)
            resp.raise_for_status()
            try:
                return resp.json()
            except ValueError as ve:
                logger.log("fetch.parse.error", "JSON parse error", {"url": url, "status": resp.status_code, "text_snippet": (resp.text[:800] if resp.text else ""), "error": str(ve)})
                return None
        except requests.exceptions.Timeout as te:
            logger.log("fetch.http.timeout", "Timeout", {"url": url, "error": str(te)})
            return None
        except requests.exceptions.SSLError as se:
            logger.log("fetch.http.sslerror", "SSL error", {"url": url, "error": str(se)})
            return None
        except requests.exceptions.RequestException as re:
            logger.log("fetch.http.requesterror", "RequestException", {"url": url, "error": str(re)})
            return None
        except Exception as e:
            logger.log("fetch.http.unexpected", "Unexpected error", {"url": url, "error": str(e), "trace": traceback.format_exc()})
            return None

    # 1) Try IP-based endpoint first
    try:
        result = try_get(ip_url, params=params, verify=True, timeout_local=timeout)
        if result:
            logger.log("fetch.agent.success", "Fetched agent details via IP", {"agent_id": agent_id, "base": "ip"})
            return result
    except Exception as e:
        logger.log("fetch.agent.error", "IP attempt raised", {"agent_id": agent_id, "error": str(e), "trace": traceback.format_exc()})

    # 2) Try domain endpoint normally (TLS verify)
    try:
        result = try_get(domain_url, params=params, verify=True, timeout_local=timeout)
        if result:
            logger.log("fetch.agent.success", "Fetched agent details via domain (verified TLS)", {"agent_id": agent_id, "base": "domain_verified"})
            return result
    except Exception as e:
        logger.log("fetch.agent.error", "Domain verified attempt raised", {"agent_id": agent_id, "error": str(e), "trace": traceback.format_exc()})

    # 3) Try domain endpoint with verify=False as diagnostic (not recommended long term)
    try:
        result = try_get(domain_url, params=params, verify=False, timeout_local=timeout)
        if result:
            logger.log("fetch.agent.success", "Fetched agent details via domain (TLS verify disabled)", {"agent_id": agent_id, "base": "domain_no_verify"})
            return result
        else:
            logger.log("fetch.agent.info", "Domain no-verify returned no JSON", {"agent_id": agent_id})
    except Exception as e:
        logger.log("fetch.agent.error", "Domain no-verify attempt raised", {"agent_id": agent_id, "error": str(e), "trace": traceback.format_exc()})

    # 4) TCP checks and subprocess curl as diagnostics
    diagnostics = {}
    try:
        diagnostics['tcp_checks'] = {}
        for host, port in [("54.89.185.235", 8000), ("api.rhythmflow.ai", 443)]:
            try:
                s = socket.create_connection((host, port), timeout=5)
                s.close()
                diagnostics['tcp_checks'][f"{host}:{port}"] = "ok"
            except Exception as e:
                diagnostics['tcp_checks'][f"{host}:{port}"] = f"fail: {type(e).__name__} {str(e)}"
    except Exception as e:
        diagnostics['tcp_error'] = str(e)

    try:
        cmd = f"curl -sS -m 20 'http://54.89.185.235:8000/fetch?table=agents&id={agent_id}'"
        diagnostics['curl_ip_cmd'] = cmd
        p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ)
        out, err = p.communicate(timeout=25)
        diagnostics['curl_ip_exit'] = p.returncode
        diagnostics['curl_ip_out'] = (out.decode('utf-8', errors='replace')[:2000] if out else "")
        diagnostics['curl_ip_err'] = (err.decode('utf-8', errors='replace')[:2000] if err else "")
    except Exception as e:
        diagnostics['curl_error'] = str(e)

    logger.log("fetch.diagnostics", "All fetch attempts failed; diagnostics attached", {"agent_id": agent_id, "diagnostics": diagnostics})
    return None

def fetch_agent_tools(agent_user_id: str, timeout: int = 8) -> Optional[Dict[str, Any]]:
    """
    Simplified and reliable tools fetcher.

    Strategy:
    1. Try primary IP-based endpoint (fast, no TLS ambiguity).
    2. Fallback once to domain-based HTTPS endpoint.
    3. Fail fast with clear logs.
    """

    headers = {
        "User-Agent": "rhythmflow-agent/1.0",
        "Accept": "application/json"
    }

    params = {
        "table": "all-agents-tools",
        "agent_id": agent_user_id
    }

    endpoints = [
        FETCH_BASE,
        DOMAIN_FALLBACK
    ]

    for url in endpoints:
        try:
            logger.log("fetch.tools.request", "Fetching agent tools", {
                "agent_user_id": agent_user_id,
                "url": url
            })

            resp = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout
            )

            if resp.status_code != 200:
                logger.log("fetch.tools.http_error", "Non-200 response", {
                    "url": url,
                    "status": resp.status_code,
                    "snippet": resp.text[:300]
                })
                continue

            data = resp.json()
            if not isinstance(data, dict):
                logger.log("fetch.tools.invalid_payload", "Expected JSON object", {
                    "url": url
                })
                continue

            logger.log("fetch.tools.success", "Agent tools fetched", {
                "agent_user_id": agent_user_id,
                "tool_count": len(data)
            })
            return data

        except requests.exceptions.Timeout:
            logger.log("fetch.tools.timeout", "Request timed out", {
                "url": url
            })

        except Exception as e:
            logger.log("fetch.tools.exception", "Unexpected error", {
                "url": url,
                "error": str(e)
            })

    logger.log("fetch.tools.failed", "Failed to fetch agent tools from all endpoints", {
        "agent_user_id": agent_user_id
    })
    return None


# ---------------------------
# Dynamic Tool Factory
# ---------------------------
def create_universal_tools(config: Dict[str, Any]) -> List[StructuredTool]:
    generated_tools = []

    for tool_id, cfg in config.items():
        api_url = cfg.get("api_url", "")
        instructions = cfg.get("instructions", "")
        condition = cfg.get("when_run", cfg.get("when_to_run", ""))
        raw_payload_template = urllib.parse.unquote(cfg.get("api_payload_json", ""))

        # Use defaults in closure to avoid late-binding issues
        def make_api_call(tool_input: Any, url=api_url, tid=tool_id) -> str:
            event_id = str(uuid.uuid4())
            payload = {}

            # Robust parsing (handles strings or dicts)
            if isinstance(tool_input, str):
                try:
                    cleaned = tool_input.strip().strip('`').replace('json\n', '', 1)
                    payload = json.loads(cleaned)
                except Exception:
                    return f"Error: Invalid JSON format."
            else:
                payload = tool_input

            logger.log("tool.call", f"api_tool_{tid} triggered", {"event_id": event_id, "payload": payload})
            
            try:
                resp = requests.post(url, json=payload, timeout=15)
                resp.raise_for_status()
                return f"Success: Data synced to API."
            except Exception as e:
                return f"API Call failed: {str(e)}"

        tool_desc = (
            f"CRITICAL CONDITION: {condition}. "
            f"INSTRUCTIONS: {instructions}. "
            f"REQUIRED JSON STRUCTURE: {raw_payload_template}"
        )

        new_tool = StructuredTool.from_function(
            func=make_api_call,
            name=f"sync_data_tool_{tool_id}",
            description=tool_desc
        )
        generated_tools.append(new_tool)
    
    return generated_tools

# ---------------------------
# Core Execution logic
# ---------------------------
def run_agent(agent_id: str, conversation_history: List[Dict[str, Any]], message: str):
    """
    New behavior:
    - Accepts agent_id, conversation_history, message
    - Fetch agent details from /fetch?table=agents&id=<agent_id>
    - Use agent response's prompt as system prompt
    - Fetch agent tools using agent_resp['user_id'] and merge them into the dynamic tool config
    - Provider is always openai. The API key is taken from agent details if present, else from environment variable OPENAI_API_KEY
    """
    logger.clear()
    logger.log("run.start", "Agent started", {"input_agent_id": agent_id})

    # 1. Fetch agent details
    agent_resp = fetch_agent_details(agent_id)
    if not agent_resp:
        logger.log("run.error", "Agent details fetch returned None", {"agent_id": agent_id})
        return {"reply": "Error: Failed to fetch agent details.", "logs": logger.to_list()}

    # Determine the api key to use
    # Prefer an API key stored in the agent details under common keys, otherwise fallback to env var.
    api_key_to_use = None
    for key_name in ("api_key", "openai_api_key", "openai_key", "key"):
        if key_name in agent_resp and agent_resp.get(key_name):
            api_key_to_use = agent_resp.get(key_name)
            break
    if not api_key_to_use:
        api_key_to_use = os.getenv("OPENAI_API_KEY")

    # Get agent prompt (instruction)
    agent_prompt = agent_resp.get("prompt", "You are a helpful assistant.")

    # 2. Fetch tools associated with this agent
    # The sample uses agent_user_id "1" which maps to agent_resp['user_id']
    agent_user_id = agent_resp.get("user_id", agent_id)
    fetched_tools = fetch_agent_tools(agent_user_id)
    merged_config = DYNAMIC_CONFIG.copy()

    # If fetched_tools is a mapping of ids to tool definitions, merge them
    if isinstance(fetched_tools, dict):
        for tid, tcfg in fetched_tools.items():
            try:
                # Ensure tool id is string
                str_tid = str(tid)
                merged_config[str_tid] = {
                    "api_url": tcfg.get("api_url", ""),
                    "api_payload_json": tcfg.get("api_payload_json", ""),
                    "instructions": tcfg.get("instructions", ""),
                    "when_run": tcfg.get("when_run", tcfg.get("when_run", "")),
                }
            except Exception as e:
                logger.log("merge.tool.error", "Failed to merge tool", {"tool_id": tid, "error": str(e)})
    else:
        logger.log("run.tools", "No tools fetched, using default config", {})

    logger.log("run.config", "Merged dynamic config", {"tool_count": len(merged_config)})

    # 3. Create tools
    tools = create_universal_tools(merged_config)

    try:
        # 4. Select Model (provider is always openai as required)
        llm = ChatOpenAI(api_key=api_key_to_use, model="gpt-4o-mini", temperature=0)

        # 5. Define modern Tool-Calling Prompt using agent_prompt
        system_message = (
            f"{agent_prompt}\n\n"
            "You have tools available for data syncing. "
            "CRITICAL RULE: Only use a tool if the user's request matches the tool's 'CRITICAL CONDITION'. "
            "If the user is asking general questions, do NOT use any tools. Just answer naturally."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # 6. Create Agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True
        )

        # 7. Format History
        history = []
        for turn in conversation_history:
            if turn.get("role") == "user":
                history.append(("human", turn.get("content")))
            else:
                history.append(("ai", turn.get("content")))

        # 8. Execute
        response = executor.invoke({
            "input": message,
            "chat_history": history
        })

        return {
            "reply": response.get("output", ""), 
            "logs": logger.to_list()
        }

    except Exception as e:
        logger.log("run.error", str(e), {"traceback": traceback.format_exc()})
        return {"reply": f"Error: {str(e)}", "logs": logger.to_list()}

# ---------------------------
# API Wrapper
# ---------------------------
app = FastAPI()

@app.post("/run-agent")
async def run_endpoint(request: Request):
    """
    Expected request JSON:
    {
        "agent_id": "agent1",
        "message": "Hello",
        "conversation": [ {"role":"user","content":"..."} ]
    }
    The code will fetch the agent details and tools, then run the agent using the agent's prompt and tools.
    """
    body = await request.json()
    agent_id = body.get("agent_id") or body.get("agentId") or body.get("id")
    if not agent_id:
        return JSONResponse({"reply": "Error: Missing agent_id in request", "logs": logger.to_list()})

    message = body.get("message", "")
    conversation = body.get("conversation", [])

    res = run_agent(agent_id, conversation, message)
    return JSONResponse(res)

@app.get("/diag")
async def diagnostics():
    """
    Diagnostic endpoint to run inside the FastAPI process.
    Call: curl -sS http://localhost:8000/diag
    It returns JSON with env proxy variables, TCP connect checks, requests attempts, and a subprocess curl attempt.
    """
    out = {"env": {}, "tcp": {}, "requests": {}, "curl_subprocess": {}, "logger": logger.to_list()}

    # 1) Environment proxy variables and important env
    for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy"):
        out["env"][k] = os.environ.get(k)

    # 2) TCP connect checks
    hosts = [("54.89.185.235", 8000), ("api.rhythmflow.ai", 443)]
    for host, port in hosts:
        key = f"{host}:{port}"
        try:
            s = socket.create_connection((host, port), timeout=5)
            s.close()
            out["tcp"][key] = "ok"
        except Exception as e:
            out["tcp"][key] = f"fail: {type(e).__name__} {str(e)}"

    # 3) requests checks using session.trust_env True
    sess = requests.Session()
    sess.trust_env = True
    headers = {"User-Agent": "diag-agent/1.0", "Accept": "application/json"}
    targets = {
        "ip_http": ("http://54.89.185.235:8000/fetch", {"table": "agents", "id": "agent1"}),
        "domain_https": ("https://api.rhythmflow.ai/fetch", {"table": "agents", "id": "agent1"})
    }
    for name, (url, params) in targets.items():
        try:
            r = sess.get(url, params=params, headers=headers, timeout=10, allow_redirects=True, verify=True)
            out["requests"][name] = {"status": r.status_code, "len": len(r.text or ""), "text_snippet": (r.text[:500] if r.text else "")}
        except Exception as e:
            out["requests"][name] = {"error": f"{type(e).__name__}: {str(e)}"}

        # try with verify=False as diagnostic
        try:
            r2 = sess.get(url, params=params, headers=headers, timeout=10, allow_redirects=True, verify=False)
            out["requests"][f"{name}_no_verify"] = {"status": getattr(r2, "status_code", None), "len": len(getattr(r2, "text", "") or ""), "text_snippet": (getattr(r2, "text", "")[:500] if getattr(r2, "text", "") else "")}
        except Exception as e:
            out["requests"][f"{name}_no_verify"] = {"error": f"{type(e).__name__}: {str(e)}"}

    # 4) Subprocess curl diagnostic (same user as FastAPI process)
    try:
        cmd = f"curl -sS -m 20 'http://54.89.185.235:8000/fetch?table=agents&id=agent1'"
        p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ)
        out_bytes, err_bytes = p.communicate(timeout=25)
        out["curl_subprocess"]["cmd"] = cmd
        out["curl_subprocess"]["exitcode"] = p.returncode
        out["curl_subprocess"]["stdout_snippet"] = out_bytes.decode("utf-8", errors="replace")[:2000]
        out["curl_subprocess"]["stderr_snippet"] = err_bytes.decode("utf-8", errors="replace")[:2000]
    except Exception as e:
        out["curl_subprocess"]["error"] = f"{type(e).__name__}: {str(e)}"

    return JSONResponse(out)

# Support for interactive runtime evaluation similar to original script
if "inputs" in globals():
    data = globals().get("inputs") or globals().get("input") or {}
    agent_id = data.get("agent_id") or data.get("agentId") or data.get("id")
    _out = run_agent(agent_id, data.get("conversation", []), data.get("message", ""))
    globals()["result"] = _out
