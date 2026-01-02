# main.py - Modern Tool Calling Agent (Dynamic Decision Making)
import os
import json
import requests
import traceback
import time
import uuid
import urllib.parse
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
import re

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

    def _agent_match(row_val: Any, target: Any) -> bool:
        """Allow matching on '3' vs 'agent3' by normalizing forms."""
        row_str = str(row_val)
        tgt_str = str(target)
        row_core = row_str[5:] if row_str.startswith("agent") else row_str
        tgt_core = tgt_str[5:] if tgt_str.startswith("agent") else tgt_str
        return row_str == tgt_str or row_core == tgt_core

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

            # normalize stored agent_id and compare as strings (accept '3' or 'agent3')
            row_agent_val = row.get("agent_id", "")
            if not _agent_match(row_agent_val, agent_user_id):
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

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_OPTIONAL_KEYS = {"description", "notes", "memo", "comments", "comment"}

def _is_empty(val: Any) -> bool:
    return val is None or (isinstance(val, str) and val.strip() == "") or (isinstance(val, (list, dict)) and len(val) == 0)

def _looks_like_email_field(name: str) -> bool:
    n = name.lower()
    return "email" in n or n in {"attendees", "emails", "invitees"}

def _normalize_email_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str) and _EMAIL_RE.match(v)]
    if isinstance(value, str) and _EMAIL_RE.match(value):
        return [value]
    return []

def _validate_and_fill_payload(payload: Dict[str, Any], template_obj: Optional[Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Validate payload essentials generically.
    - Enforce non-empty values for template keys unless clearly optional.
    - For email-like fields, require at least one valid email.
    - If both start_date_time and end_date_time are in template and end is missing, default to start+30m when parseable.
    Returns (ok, question, payload). If ok is False, question is a single user-facing ask.
    """
    missing: List[str] = []

    if isinstance(template_obj, dict):
        keys = list(template_obj.keys())
        for k in keys:
            if k in _OPTIONAL_KEYS:
                continue
            if _looks_like_email_field(k):
                emails = _normalize_email_list(payload.get(k))
                if not emails:
                    missing.append(k)
                else:
                    payload[k] = emails
                continue

            if k not in payload or _is_empty(payload.get(k)):
                missing.append(k)

        # Simple default: if both start_date_time and end_date_time exist in template and end is empty, set end = start + 30m when parseable
        if "start_date_time" in template_obj and "end_date_time" in template_obj and _is_empty(payload.get("end_date_time")):
            start_val = payload.get("start_date_time")
            if isinstance(start_val, str):
                try:
                    dt = datetime.fromisoformat(start_val)
                    payload["end_date_time"] = (dt + timedelta(minutes=30)).isoformat()
                except Exception:
                    pass

    if missing:
        if len(missing) == 1:
            return False, f"Could you share the value for '{missing[0]}' to proceed?", payload
        return False, f"Could you share the values for {', '.join(missing[:-1])} and {missing[-1]} to proceed?", payload

    return True, None, payload


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

    # 2. Fetch tools associated with this agent (Redis-only) using the agent_id from payload
    fetched_tools = fetch_agent_tools(str(agent_id))
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

        def _compact_json(obj: Any, max_chars: int = 1800) -> str:
            try:
                s = json.dumps(obj, ensure_ascii=False)
            except Exception:
                s = str(obj)
            return s if len(s) <= max_chars else (s[:max_chars] + "…")

        def _parse_ask_guidance(instructions_text: str) -> Dict[str, str]:
            """
            Extract AskGuidance per field path from tool instructions.
            Supports keys like 'title' and 'values.A'.
            """
            out: Dict[str, str] = {}
            if not instructions_text:
                return out
            for m in re.finditer(r"'([^']+)'.*?AskGuidance=([A-Z_]+)", instructions_text):
                field_path = m.group(1).strip()
                guidance = m.group(2).strip()
                if field_path:
                    out[field_path] = guidance
            return out

        def _build_tool_specs() -> List[Dict[str, Any]]:
            specs: List[Dict[str, Any]] = []
            for tid, cfg in (merged_config.items() if isinstance(merged_config, dict) else []):
                name = f"sync_data_tool_{tid}"
                tpl_raw = urllib.parse.unquote(str(cfg.get("api_payload_json", "") or ""))
                tpl_obj = _safe_json_loads(tpl_raw)
                instr = str(cfg.get("instructions", "") or "")
                ask_map = _parse_ask_guidance(instr)
                specs.append({
                    "name": name,
                    "when_run": str(cfg.get("when_run", "") or "").strip(),
                    "payload_template": tpl_obj if tpl_obj is not None else tpl_raw,
                    "ask_guidance": ask_map,
                    "instructions": (instr[:800] + "…") if len(instr) > 800 else instr,
                })
            return specs

        tool_specs = _build_tool_specs()

        def _tool_spec_by_name(nm: str) -> Optional[Dict[str, Any]]:
            for s in tool_specs:
                if s.get("name") == nm:
                    return s
            return None

        def _is_missing_value(v: Any) -> bool:
            return v is None or (isinstance(v, str) and v.strip() == "") or (isinstance(v, (list, dict)) and len(v) == 0)

        def _validate_payload_with_askmap(payload: Dict[str, Any], template_obj: Any, ask_map: Dict[str, str]) -> Tuple[bool, Optional[str]]:
            """
            Generic validation:
            - Payload must include all top-level keys present in template (structure match).
            - Fields with AskGuidance=SHOULD_BE_ASKED must be present and non-empty.
            - If there are email-like fields, enforce valid emails when non-empty (or when SHOULD_BE_ASKED).
            Returns (ok, question_if_missing).
            """
            if not isinstance(template_obj, dict):
                return True, None

            # Ensure structure keys exist
            for k in template_obj.keys():
                if k not in payload:
                    return False, f"Could you share the value for '{k}' to proceed?"

            # Validate required fields from AskGuidance
            missing_required: List[str] = []
            for field_path, guidance in ask_map.items():
                if guidance != "SHOULD_BE_ASKED":
                    continue
                # Only enforce paths that exist in the template (top-level or dot-path)
                parts = field_path.split(".")
                cur: Any = payload
                for p in parts:
                    if not isinstance(cur, dict) or p not in cur:
                        cur = None
                        break
                    cur = cur[p]
                if _is_missing_value(cur):
                    missing_required.append(field_path)

            # Email heuristic: if any required field looks email-like, enforce format
            invalid_emails: List[str] = []
            for field_path, guidance in ask_map.items():
                if guidance != "SHOULD_BE_ASKED":
                    continue
                if not _looks_like_email_field(field_path.split(".")[-1]):
                    continue
                parts = field_path.split(".")
                cur: Any = payload
                for p in parts:
                    if not isinstance(cur, dict) or p not in cur:
                        cur = None
                        break
                    cur = cur[p]
                emails = _normalize_email_list(cur)
                if not emails:
                    invalid_emails.append(field_path)

            if invalid_emails:
                return False, "Please share a valid email address to proceed."

            if missing_required:
                if len(missing_required) == 1:
                    return False, f"Could you share {missing_required[0]} to proceed?"
                return False, f"Could you share {', '.join(missing_required[:-1])} and {missing_required[-1]} to proceed?"

            return True, None

        # ---------------------------
        # Planner loop: decide next question or tool, execute, repeat (no retries)
        # ---------------------------
        tool_run_results: List[Dict[str, Any]] = []
        executed_tools: set[str] = set()
        failed_tools: set[str] = set()

        max_steps = 6
        for step_idx in range(max_steps):
            remaining_specs = [s for s in tool_specs if s.get("name") not in executed_tools and s.get("name") not in failed_tools]

            planner_system = (
                f"{agent_prompt}\n\n"
                "You are the runtime planner.\n"
                "Goal: help the user by asking for missing details (one question at a time) and running tools when ready.\n"
                "You MUST follow tool when_run and instructions, and match payload_template structure exactly.\n"
                "Use all available tools; plan dependencies (e.g., check availability, then book) based on prior tool results.\n"
                "Never retry a failed tool in this request.\n"
                "Do not claim an external action succeeded unless a tool result clearly confirms it.\n\n"
                "Output MUST be strict JSON with keys:\n"
                '{"action":"ask"|"run_tool"|"done","question":string|null,"tool_name":string|null,"payload":object|null,"reply":string|null,"reason":string}\n'
                "Rules:\n"
                "- If action=ask: question must be one specific question; tool_name/payload must be null.\n"
                "- If action=run_tool: tool_name and payload required; question/reply must be null.\n"
                "- If action=done: reply required; question/tool_name/payload must be null.\n"
                "No extra text."
            )

            planner_user = (
                "Conversation history:\n"
                + _compact_json(history_msgs, 2200)
                + "\n\nUser message:\n"
                + (message or "")
                + "\n\nAvailable tools (compact):\n"
                + _compact_json(remaining_specs, 2600)
                + "\n\nPrior tool runs (compact):\n"
                + _compact_json(tool_run_results, 2200)
            )

            plan_raw = llm.invoke(
                [{"role": "system", "content": planner_system}, {"role": "user", "content": planner_user}],
                response_format={"type": "json_object"},
            ).content

            plan_clean = _strip_code_fences(plan_raw or "")
            plan_obj = _safe_json_loads(plan_clean) if plan_clean else None
            if not isinstance(plan_obj, dict):
                logger.log("planner.raw", "Planner parse failed", {"raw": plan_raw})
                return {"reply": "Sorry — I couldn't understand the next step. Could you rephrase?", "logs": logger.to_list()}

            action = str(plan_obj.get("action") or "").strip()
            logger.log("planner.step", "Planner step", {"step": step_idx, "action": action, "reason": plan_obj.get("reason", "")})

            if action == "ask":
                q = plan_obj.get("question")
                q = str(q).strip() if q is not None else ""
                if not q:
                    q = "Could you share the missing details needed to proceed?"
                return {"reply": q, "logs": logger.to_list()}

            if action == "done":
                reply = plan_obj.get("reply")
                reply = str(reply).strip() if reply is not None else ""
                if not reply:
                    reply = "Done."
                return {"reply": reply, "logs": logger.to_list()}

            if action != "run_tool":
                return {"reply": "Sorry — I couldn't determine the next step. Could you rephrase?", "logs": logger.to_list()}

            tool_name = plan_obj.get("tool_name")
            tool_name = str(tool_name).strip() if tool_name is not None else ""
            payload = plan_obj.get("payload")

            if not tool_name or tool_name not in tools_by_name:
                return {"reply": "Error: Invalid tool selected.", "logs": logger.to_list()}
            if tool_name in executed_tools or tool_name in failed_tools:
                return {"reply": "Error: Tool already used in this request.", "logs": logger.to_list()}
            if not isinstance(payload, dict):
                return {"reply": "Error: Tool payload must be an object.", "logs": logger.to_list()}

            spec = _tool_spec_by_name(tool_name) or {}
            tpl_obj = spec.get("payload_template")
            ask_map = spec.get("ask_guidance") or {}
            if isinstance(tpl_obj, str):
                tpl_obj_parsed = _safe_json_loads(tpl_obj)
                tpl_obj = tpl_obj_parsed if tpl_obj_parsed is not None else tpl_obj

            # Validate structure and required fields
            ok, ask_q = _validate_payload_with_askmap(payload, tpl_obj, ask_map)
            if not ok:
                return {"reply": ask_q or "Could you share the missing details needed to proceed?", "logs": logger.to_list()}

            # Execute exactly once
            logger.log("tool.payload_plan", "Payload plan prepared", {"tool_name": tool_name, "should_call": True, "reason": plan_obj.get("reason", "")})
            chosen_tool = tools_by_name[tool_name]
            try:
                tool_result_raw = chosen_tool.invoke(payload)
            except Exception as e:
                logger.log("tool.error", "Tool invocation error", {"tool_name": tool_name, "error": str(e), "traceback": traceback.format_exc()})
                tool_result_raw = json.dumps({"ok": False, "error": str(e), "status_code": None, "response": None, "event_id": str(uuid.uuid4())})

            tool_result_text = tool_result_raw if isinstance(tool_result_raw, str) else json.dumps(tool_result_raw)
            tool_result_obj = _safe_json_loads(tool_result_text) if isinstance(tool_result_text, str) else None
            if not isinstance(tool_result_obj, dict):
                tool_result_obj = {"ok": False, "error": "Tool returned non-JSON result", "raw": tool_result_text}

            tool_run_results.append({"tool_name": tool_name, "payload": payload, "result": tool_result_obj})
            executed_tools.add(tool_name)
            if not bool(tool_result_obj.get("ok")):
                failed_tools.add(tool_name)

            # Continue loop to allow dependent follow-up tools in the same request.
            continue

        # If we hit max steps, produce a final response based on what we did.
        fallback_reply = "I’ve completed the available steps for now. What would you like to do next?"
        return {"reply": fallback_reply, "logs": logger.to_list()}

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
