# main.py - Modern Tool Calling Agent (Dynamic Decision Making)
import os
import json
import requests
import traceback
import time
import uuid
import urllib.parse
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
# Helper: fetch agent details and tools from Redis-only
# ---------------------------
# Note: we intentionally use Redis only. No HTTP fallback.

import os
from upstash_redis import Redis as UpstashRedis

_redis_client = None

try:
    # 1. Pull credentials from environment variables (Railway)
    # Falling back to the provided hardcoded values if environment variables are missing
    redis_url = os.getenv("UPSTASH_REDIS_REST_URL", "https://climbing-hyena-56303.upstash.io")
    redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "AdvvAAIncDExZmMzYTBiNTJhZWU0MzA1YjA1M2IwYWU4NThlZjcyM3AxNTYzMDM")
    
    if not redis_url or not redis_token:
        raise RuntimeError("Upstash credentials missing in environment")

    # 2. Initialize the HTTP-based Upstash client
    _redis_client = UpstashRedis(url=redis_url, token=redis_token)
    
    try:
        # 3. Test connection (Upstash uses .set or .get to verify HTTP connectivity)
        _redis_client.set("connection_test", "ok")
        logger.log("redis", "Connected to Upstash via HTTP SDK", {"url": redis_url})
        print("✅ Connected to Upstash via HTTP SDK")
    except Exception as e:
        logger.log("redis.ping_error", "Upstash test operation failed", {"error": str(e)})
        _redis_client = None
        
except Exception as e:
    logger.log("redis.init_error", "Failed to initialize Upstash client", {"error": str(e)})
    print(f"❌ Upstash SDK connection failed: {e}")
    _redis_client = None

# For compatibility with the rest of your app logic
r = _redis_client

def fetch_agent_details(agent_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch agent details from Redis only.
    agent_id may be 'agent1' or '1'. We check both forms.
    Returns a dict of the agent hash or None if not found.
    """
    if not _redis_client:
        logger.log("fetch.agent.no_redis", "Redis client not available", {"agent_id": agent_id})
        return None

    try:
        # build candidate keys to try
        candidates = []
        if isinstance(agent_id, str):
            candidates.append(agent_id)
            if agent_id.startswith("agent"):
                # keep as-is, but also try numeric suffix
                try:
                    suffix = int(agent_id[len("agent"):])
                    candidates.append(str(suffix))
                    candidates.append(f"agent{suffix}")
                except Exception:
                    pass
            else:
                # try numeric and agent<id>
                candidates.append(f"agent{agent_id}")
        else:
            candidates.append(f"agent{agent_id}")
            candidates.append(str(agent_id))

        seen = set()
        for cand in candidates:
            if not cand or cand in seen:
                continue
            seen.add(cand)
            # prefer actual agent key like "agent1"
            key = cand if cand.startswith("agent") else f"agent{cand}" if cand.isdigit() else cand
            try:
                if _redis_client.exists(key):
                    rec = _redis_client.hgetall(key) or {}
                    # parse tools field if present
                    if "tools" in rec and rec.get("tools"):
                        try:
                            rec["tools"] = json.loads(rec["tools"])
                        except Exception:
                            # keep as-is if not JSON
                            pass
                    # normalize numeric timestamps
                    for fld in ("created_at", "updated_at"):
                        if fld in rec:
                            try:
                                rec[fld] = int(rec[fld])
                            except Exception:
                                pass
                    logger.log("fetch.agent.redis", "Fetched agent details from Redis", {"agent_key": key})
                    return rec
            except Exception as e:
                # continue trying other candidates but log the error
                logger.log("fetch.agent.redis.error", "Error checking agent key in Redis", {"candidate": key, "error": str(e)})
                continue

        # Not found
        logger.log("fetch.agent.redis.notfound", "Agent not found in Redis", {"agent_id": agent_id, "candidates_tried": list(seen)})
        return None

    except Exception as e:
        logger.log("fetch.agent.redis.exception", "Unexpected error fetching agent", {"agent_id": agent_id, "error": str(e)})
        return None

def fetch_agent_tools(agent_user_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch all rows from dynamic table 'all-agents-tools' directly from Redis only.
    Filter rows where row['agent_id'] == agent_user_id (string comparison).
    Return a dict of tool_id -> tool_definition or None if Redis unavailable.
    """
    if not _redis_client:
        logger.log("fetch.tools.no_redis", "Redis client not available", {"agent_user_id": agent_user_id})
        return None

    try:
        table_name = "all-agents-tools"
        ids_key = f"table:{table_name}:ids"
        row_ids = _redis_client.smembers(ids_key) or set()
        tool_map: Dict[str, Any] = {}

        for row_id in row_ids:
            if row_id == "_meta":
                continue
            row_key = f"table:{table_name}:row:{row_id}"
            try:
                row = _redis_client.hgetall(row_key) or {}
            except Exception as e:
                logger.log("fetch.tools.row.error", "Failed to hgetall for row", {"row_key": row_key, "error": str(e)})
                continue

            if not row:
                continue

            # normalize stored agent_id and compare as strings
            row_agent_val = row.get("agent_id", "")
            if str(row_agent_val) != str(agent_user_id):
                continue

            tool_map[str(row_id)] = {
                "api_url": row.get("api_url", "") or "",
                "api_payload_json": row.get("api_payload_json", "") or "",
                "instructions": row.get("instructions", "") or "",
                "when_run": row.get("when_run", "") or ""
            }

        logger.log("fetch.tools.redis", "Fetched tools from Redis", {"agent_user_id": agent_user_id, "count": len(tool_map)})
        return tool_map

    except Exception as e:
        logger.log("fetch.tools.redis.exception", "Error fetching tools from Redis", {"agent_user_id": agent_user_id, "error": str(e)})
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

        # We define the function to take ANY named arguments
        def make_api_call(url=api_url, tid=tool_id, **kwargs) -> str:
            event_id = str(uuid.uuid4())
            
            # If the LLM sends 'prompt', it will be in kwargs.
            # If LangChain wraps everything in 'tool_input', we grab that.
            payload = kwargs.get("tool_input", kwargs) if kwargs else {}
            
            # Clean up internal keys so they don't go to ActivePieces
            payload_to_send = {k: v for k, v in payload.items() if k not in ["url", "tid"]}

            logger.log("tool.call", f"api_tool_{tid} triggered", {"event_id": event_id, "payload": payload_to_send})

            try:
                resp = requests.post(url, json=payload_to_send, timeout=15)
                status_code = resp.status_code
                
                # Logic to handle response
                try:
                    res_data = resp.json()
                except:
                    res_data = resp.text

                ok = 200 <= status_code < 300
                
                tool_result = {
                    "ok": ok,
                    "tool_id": str(tid),
                    "status_code": status_code,
                    "response": res_data
                }
                logger.log("tool.response", f"api_tool_{tid} success", tool_result)
                return json.dumps(tool_result)

            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        # IMPORTANT: We tell the tool explicitly what the input schema looks like
        # This forces LangChain to map 'prompt' into the kwargs
        new_tool = StructuredTool.from_function(
            func=make_api_call,
            name=f"sync_data_tool_{tool_id}",
            description=(
                f"USE CASE: {condition}. "
                f"REQUIRED JSON FIELDS: {raw_payload_template}. "
                f"INSTRUCTIONS: {instructions}"
            )
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
    - Fetch agent details from Redis-only
    - Use agent response's prompt as system prompt
    - Fetch agent tools from Redis-only using agent_resp['user_id'] and merge them into dynamic tool config
    - Provider is always openai. The API key is taken from agent details if present, else from environment variable OPENAI_API_KEY

    Tool behavior:
    - Tool calls are synchronous and blocking.
    - Each tool returns the real API response as JSON (string) to the agent.
    - Each tool response is also logged as "tool.response" so your API output includes it.
    """
    logger.clear()
    logger.log("run.start", "Agent started", {"input_agent_id": agent_id})

    # 1. Fetch agent details (Redis-only)
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

    # 2. Fetch tools associated with this agent (Redis-only)
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
        logger.log("run.tools", "No tools fetched from Redis, using default config", {})

    logger.log("run.config", "Merged dynamic config", {"tool_count": len(merged_config)})

    # 3. Create tools
    tools = create_universal_tools(merged_config)

    try:
        # 4. Select Model (provider is always openai as required)
        llm = ChatOpenAI(api_key=api_key_to_use, model="gpt-4o-mini", temperature=0)

        # 5. Build a list of tool "when_run" values so the agent knows what it can do
        _conditions_raw: List[str] = []
        for _tid, _cfg in merged_config.items():
            _cond = _cfg.get("when_run", _cfg.get("when_to_run", ""))
            if _cond:
                _conditions_raw.append(str(_cond).strip())

        _seen_conditions = set()
        _conditions: List[str] = []
        for _c in _conditions_raw:
            if _c and _c not in _seen_conditions:
                _seen_conditions.add(_c)
                _conditions.append(_c)

        if _conditions:
            tools_when_run_text = "Tool conditions you can act on (when_run):\n" + "\n".join(
                [f"{i+1}) {c}" for i, c in enumerate(_conditions)]
            )
        else:
            tools_when_run_text = "Tool conditions you can act on (when_run): None."

        # 6. Define modern Tool-Calling Prompt using agent_prompt and the tools' when_run list
        system_message = (
            f"{agent_prompt}\n\n"
            f"{tools_when_run_text}\n\n"
            "CRITICAL RULE: Only use a tool if the user's request matches the relevant when_run condition above. "
            "When you call tools, you MUST rely on the tool's returned JSON to determine whether the action succeeded. "
            "If any tool returns ok=false, you must report the failure and the error. "
            "If tools return ok=true, you may confirm success using details from response_json or response_text. "
            "If the user is asking general questions, do NOT use any tools. Just answer naturally."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # 7. Create Agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True
        )

        # 8. Format History (be tolerant to both "content" and "message")
        history = []
        for turn in conversation_history:
            role = (turn.get("role") or "").lower().strip()
            content = turn.get("content")
            if content is None:
                content = turn.get("message")
            if content is None:
                content = ""
            # Accept common role variants
            if role == "user":
                history.append(("human", content))
            else:
                # Treat agent/assistant/ai as ai
                history.append(("ai", content))

        # 9. Execute
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

# Support for interactive runtime evaluation similar to original script
if "inputs" in globals():
    data = globals().get("inputs", {})
    agent_id = data.get("agent_id") or data.get("agentId") or data.get("id")
    _out = run_agent(agent_id, data.get("conversation", []), data.get("message", ""))
    globals()["result"] = _out
