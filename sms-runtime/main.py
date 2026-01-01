# main.py - Modern Tool Calling Agent (Dynamic Decision Making)
import os
import json
import requests
import traceback
import time
import uuid
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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

}

# ---------------------------
# Logger helper
# ---------------------------
class Logger:
    def __init__(self):
        self._events: List[Dict[str, Any]] = []
    def _now(self) -> str:
        # timezone-aware UTC timestamp to avoid datetime.utcnow() deprecation
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
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
from pydantic import create_model, Field

def _safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None

def _strip_code_fences(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip()
    if t.startswith("```"):
        t = t.strip("`")
        # common patterns: ```json ... ```
        t = t.replace("json\n", "", 1).replace("json\r\n", "", 1)
    return t.strip()

def create_universal_tools(config: Dict[str, Any]) -> List[StructuredTool]:
    generated_tools = []

    for tool_id, cfg in config.items():
        api_url = cfg.get("api_url", "")
        instructions = cfg.get("instructions", "")
        condition = cfg.get("when_run", "")
        raw_payload_template = urllib.parse.unquote(cfg.get("api_payload_json", ""))
        
        # 1. Try to parse the keys from your template to create a schema
        try:
            example_json = json.loads(raw_payload_template)
            # Create a dynamic Pydantic model so the LLM "sees" the fields
            # We treat all keys in your template as required fields for the LLM
            fields = {
                key: (Any, Field(description=f"Value for {key}")) 
                for key in example_json.keys()
            }
            DynamicArgsModel = create_model(f"Args_{tool_id}", **fields)
        except Exception:
            DynamicArgsModel = None

        # 2. The function now only takes kwargs (which the LLM will fill based on the schema)
        # IMPORTANT: bind tool_id and api_url as defaults to avoid late-binding bugs.
        def _make_api_call_factory(_tool_id: str, _api_url: str):
            def make_api_call(**kwargs) -> str:
                event_id = str(uuid.uuid4())
                payload = kwargs # The LLM will now fill this with 'title', 'start_date_time', etc.

                logger.log("tool.call", f"api_tool_{_tool_id} triggered", {"event_id": event_id, "payload": payload})

                try:
                    resp = requests.post(_api_url, json=payload, timeout=15)
                    try:
                        response_data = resp.json()
                    except Exception:
                        response_data = resp.text

                    tool_result = {
                        "ok": resp.ok,
                        "status_code": resp.status_code,
                        "response": response_data,
                        "event_id": event_id
                    }
                    logger.log("tool.response", f"api_tool_{_tool_id} result", tool_result)
                    return json.dumps(tool_result)

                except Exception as e:
                    error_data = {"ok": False, "error": str(e), "event_id": event_id}
                    logger.log("tool.error", f"api_tool_{_tool_id} failed", error_data)
                    return json.dumps(error_data)
            return make_api_call

        make_api_call = _make_api_call_factory(str(tool_id), str(api_url))

        # 3. Create the tool with the args_schema
        new_tool = StructuredTool.from_function(
            func=make_api_call,
            name=f"sync_data_tool_{tool_id}",
            description=f"Condition: {condition}. Instructions: {instructions}. REQUIRED JSON STRUCTURE: {raw_payload_template}",
            args_schema=DynamicArgsModel # <--- THIS IS THE MAGIC KEY
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

    Tool behavior (UPDATED):
    - Only ONE tool may be executed per run-agent call.
    - The system chooses a tool from scratch each run based on when_run conditions.
    - If a tool fails, the agent must not try other tools in the same run.
    - Payload selection is performed fresh for the chosen tool using its instructions and payload template.
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
    tools_by_name: Dict[str, StructuredTool] = {t.name: t for t in tools}

    try:
        # 4. Select Model (provider is always openai as required)
        llm = ChatOpenAI(api_key=api_key_to_use, model="gpt-4o-mini", temperature=0)

        # 5. Format History (be tolerant to both "content" and "message")
        history_msgs = []
        for turn in conversation_history:
            role = (turn.get("role") or "").lower().strip()
            content = turn.get("content")
            if content is None:
                content = turn.get("message")
            if content is None:
                content = ""
            if role == "user":
                history_msgs.append({"role": "user", "content": content})
            else:
                history_msgs.append({"role": "assistant", "content": content})

        # 6. Build a concise tool registry for the LLM to choose from
        tool_registry_lines: List[str] = []
        for _tid, _cfg in merged_config.items():
            _name = f"sync_data_tool_{_tid}"
            _when = str(_cfg.get("when_run", "") or "").strip()
            _instr = str(_cfg.get("instructions", "") or "").strip()
            _tpl = urllib.parse.unquote(str(_cfg.get("api_payload_json", "") or ""))
            tool_registry_lines.append(
                f"- name={_name}\n  when_run={_when}\n  instructions={_instr}\n  payload_template={_tpl}"
            )
        tool_registry_text = "\n".join(tool_registry_lines) if tool_registry_lines else "(no tools)"

        # ---------------------------
        # STEP A: Decide the single best tool (or no tool)
        # ---------------------------
        decision_system = (
            f"{agent_prompt}\n\n"
            "You must choose AT MOST ONE tool for this request.\n"
            "If a tool is chosen and it fails, you MUST NOT try any other tools in this same run.\n"
            "If no tool matches, choose no_tool.\n\n"
            "Available tools:\n"
            f"{tool_registry_text}\n\n"
            "Output MUST be strict JSON with keys:\n"
            '{"action":"use_tool"|"no_tool","tool_name":string|null,"reason":string}\n'
            "No extra text."
        )

        decision_user = (
            "Conversation history:\n"
            + json.dumps(history_msgs, ensure_ascii=False)
            + "\n\nUser message:\n"
            + (message or "")
        )

        decision_raw = llm.invoke([
            {"role": "system", "content": decision_system},
            {"role": "user", "content": decision_user},
        ]).content

        decision_clean = _strip_code_fences(decision_raw or "")
        decision_obj = _safe_json_loads(decision_clean) if decision_clean else None
        if not isinstance(decision_obj, dict):
            decision_obj = {"action": "no_tool", "tool_name": None, "reason": "Decision parsing failed; defaulting to no_tool."}

        action = str(decision_obj.get("action") or "no_tool").strip()
        chosen_tool_name = decision_obj.get("tool_name")
        if chosen_tool_name is not None:
            chosen_tool_name = str(chosen_tool_name).strip()

        logger.log("tool.decide", "Tool decision made", {"action": action, "tool_name": chosen_tool_name, "reason": decision_obj.get("reason", "")})

        # If no tool, answer normally (no tools)
        if action != "use_tool" or not chosen_tool_name or chosen_tool_name not in tools_by_name:
            answer_system = agent_prompt
            answer_user = (
                "Conversation history:\n"
                + json.dumps(history_msgs, ensure_ascii=False)
                + "\n\nUser message:\n"
                + (message or "")
            )
            answer = llm.invoke([
                {"role": "system", "content": answer_system},
                {"role": "user", "content": answer_user},
            ]).content
            return {"reply": answer or "", "logs": logger.to_list()}

        # ---------------------------
        # STEP B: For the chosen tool only, construct payload OR ask a single question
        # ---------------------------
        # Extract tool_id from "sync_data_tool_<id>"
        tool_id_part = chosen_tool_name.replace("sync_data_tool_", "", 1).strip()
        chosen_cfg = merged_config.get(tool_id_part, {}) if isinstance(merged_config, dict) else {}
        chosen_instructions = str(chosen_cfg.get("instructions", "") or "")
        chosen_when_run = str(chosen_cfg.get("when_run", "") or "")
        chosen_payload_template_raw = urllib.parse.unquote(str(chosen_cfg.get("api_payload_json", "") or ""))
        chosen_payload_template_obj = _safe_json_loads(chosen_payload_template_raw)

        payload_builder_system = (
            f"{agent_prompt}\n\n"
            "You are preparing inputs for exactly ONE tool call.\n"
            "You MUST follow the chosen tool's instructions including AskGuidance and AskNote.\n"
            "SHOULD_BE_ASKED means: if missing or uncertain, ask the user ONE precise question and do NOT call the tool.\n"
            "NOT_TO_BE_ASKED means: never ask, derive or set safe defaults if possible.\n"
            "CAN_BE_ASKED means: derive first, ask only if needed for correctness.\n\n"
            "You may NOT switch tools. Only prepare for the chosen tool below.\n\n"
            f"Chosen tool: {chosen_tool_name}\n"
            f"when_run: {chosen_when_run}\n"
            f"instructions: {chosen_instructions}\n"
            f"payload_template: {chosen_payload_template_raw}\n\n"
            "Output MUST be strict JSON with keys:\n"
            '{"should_call":true|false,"payload":object|null,"question":string|null,"reason":string}\n'
            "Rules:\n"
            "- If should_call=false, question MUST be a single question to the user.\n"
            "- If should_call=true, payload MUST match the template structure.\n"
            "No extra text."
        )

        payload_builder_user = (
            "Conversation history:\n"
            + json.dumps(history_msgs, ensure_ascii=False)
            + "\n\nUser message:\n"
            + (message or "")
        )

        pb_raw = llm.invoke([
            {"role": "system", "content": payload_builder_system},
            {"role": "user", "content": payload_builder_user},
        ]).content

        pb_clean = _strip_code_fences(pb_raw or "")
        pb_obj = _safe_json_loads(pb_clean) if pb_clean else None
        if not isinstance(pb_obj, dict):
            pb_obj = {"should_call": False, "payload": None, "question": "Could you share the missing details needed to proceed?", "reason": "Payload parsing failed."}

        should_call = bool(pb_obj.get("should_call"))
        question = pb_obj.get("question")
        if question is not None:
            question = str(question).strip()

        payload = pb_obj.get("payload")

        logger.log("tool.payload_plan", "Payload plan prepared", {"tool_name": chosen_tool_name, "should_call": should_call, "reason": pb_obj.get("reason", "")})

        if not should_call:
            return {"reply": question or "Could you share the missing details needed to proceed?", "logs": logger.to_list()}

        if not isinstance(payload, dict):
            return {"reply": "Error: Tool payload must be an object.", "logs": logger.to_list()}

        # Optional: basic structural check against template if template parses as dict
        if isinstance(chosen_payload_template_obj, dict):
            for k in chosen_payload_template_obj.keys():
                if k not in payload:
                    return {"reply": f"Error: Missing required top-level field '{k}' for {chosen_tool_name}.", "logs": logger.to_list()}

        # ---------------------------
        # STEP C: Execute exactly ONE tool call
        # ---------------------------
        chosen_tool = tools_by_name[chosen_tool_name]

        tool_result_raw = None
        try:
            tool_result_raw = chosen_tool.invoke(payload)
        except Exception as e:
            logger.log("tool.error", "Tool invocation error", {"tool_name": chosen_tool_name, "error": str(e), "traceback": traceback.format_exc()})
            tool_result_raw = json.dumps({"ok": False, "error": str(e), "status_code": None, "response": None, "event_id": str(uuid.uuid4())})

        # Parse tool result (your tools return JSON string)
        tool_result_text = tool_result_raw if isinstance(tool_result_raw, str) else json.dumps(tool_result_raw)
        tool_result_obj = _safe_json_loads(tool_result_text) if isinstance(tool_result_text, str) else None
        if not isinstance(tool_result_obj, dict):
            tool_result_obj = {"ok": False, "error": "Tool returned non-JSON result", "raw": tool_result_text}

        ok = bool(tool_result_obj.get("ok"))

        # ---------------------------
        # STEP D: Final response grounded on tool result (NO additional tools)
        # ---------------------------
        final_system = (
            f"{agent_prompt}\n\n"
            "You have executed exactly one tool call.\n"
            "You MUST NOT call any more tools.\n"
            "Base your reply strictly on the tool result provided.\n"
        )
        final_user = (
            "User message:\n"
            + (message or "")
            + "\n\nTool used:\n"
            + chosen_tool_name
            + "\n\nTool payload:\n"
            + json.dumps(payload, ensure_ascii=False)
            + "\n\nTool result:\n"
            + json.dumps(tool_result_obj, ensure_ascii=False)
        )

        final_reply = llm.invoke([
            {"role": "system", "content": final_system},
            {"role": "user", "content": final_user},
        ]).content

        return {"reply": final_reply or "", "logs": logger.to_list()}

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
