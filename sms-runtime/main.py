
# main.py - LangGraph create_react_agent runtime with dynamic tools (Redis fetched), reply + logs
import os
import json
import uuid
import traceback
import urllib.parse
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Annotated, Callable
from operator import ior
import requests

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from upstash_redis import Redis as UpstashRedis
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState

from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI

from pydantic import create_model, Field, BaseModel, ConfigDict
from langgraph.errors import GraphRecursionError
from xml.sax.saxutils import escape
import asyncio

# Minimal dynamic config
DYNAMIC_CONFIG: Dict[str, Any] = {}

class AgentState(MessagesState):
    # This stores the dynamic variables in agent memory and merges updates
    variables: Annotated[Dict[str, str], ior]
    # Flag used by LangGraph prebuilt agents; keep default False
    is_last_step: bool = False

# Simple logger
class Logger:
    def __init__(self):
        self._events: List[Dict[str, Any]] = []

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def log(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        entry = {"ts": self._now(), "id": str(uuid.uuid4()), "type": event_type, "message": message, "data": data or {}}
        self._events.append(entry)

    def to_list(self) -> List[Dict[str, Any]]:
        return self._events.copy()

    def clear(self):
        self._events = []

logger = Logger()

# ---------------------------
# Upstash Redis init
# ---------------------------
_redis_client: Optional[UpstashRedis] = None
try:
    redis_url = "https://climbing-hyena-56303.upstash.io"
    redis_token = "AdvvAAIncDExZmMzYTBiNTJhZWU0MzA1YjA1M2IwYWU4NThlZjcyM3AxNTYzMDM"

    # If you have local dev defaults, set them here, but never keep production secrets hardcoded
    if not redis_url or not redis_token:
        raise RuntimeError("Upstash credentials missing in environment")

    _redis_client = UpstashRedis(url=redis_url, token=redis_token)
    try:
        _redis_client.set("connection_test", "ok")
        logger.log("redis", "Connected to Upstash via HTTP SDK", {"url": redis_url})
        print("Connected to Upstash via HTTP SDK")
    except Exception as e:
        logger.log("redis.ping_error", "Upstash test operation failed", {"error": str(e)})
        _redis_client = None
except Exception as e:
    logger.log("redis.init_error", "Failed to initialize Upstash client", {"error": str(e)})
    print(f"Upstash SDK connection failed: {e}")
    _redis_client = None

# ---------------------------
# Redis helpers
# ---------------------------
def fetch_agent_details(agent_id: str) -> Optional[Dict[str, Any]]:
    if not _redis_client:
        logger.log("fetch.agent.no_redis", "Redis client not available", {"agent_id": agent_id})
        return None
    try:
        candidates: List[str] = []
        if isinstance(agent_id, str):
            candidates.append(agent_id)
            if agent_id.startswith("agent"):
                try:
                    suffix = int(agent_id[len("agent"):])
                    candidates.append(str(suffix))
                    candidates.append(f"agent{suffix}")
                except Exception:
                    pass
            else:
                candidates.append(f"agent{agent_id}")
        else:
            candidates.append(f"agent{agent_id}")
            candidates.append(str(agent_id))

        seen = set()
        for cand in candidates:
            if not cand or cand in seen:
                continue
            seen.add(cand)
            key = cand if cand.startswith("agent") else (f"agent{cand}" if cand.isdigit() else cand)
            try:
                if _redis_client.exists(key):
                    rec = _redis_client.hgetall(key) or {}
                    if "tools" in rec and rec.get("tools"):
                        try:
                            rec["tools"] = json.loads(rec["tools"])
                        except Exception:
                            pass
                    for fld in ("created_at", "updated_at"):
                        if fld in rec:
                            try:
                                rec[fld] = int(rec[fld])
                            except Exception:
                                pass
                    logger.log("fetch.agent.redis", "Fetched agent details from Redis", {"agent_key": key})
                    return rec
            except Exception as e:
                logger.log("fetch.agent.redis.error", "Error checking agent key in Redis", {"candidate": key, "error": str(e)})
                continue
        logger.log("fetch.agent.redis.notfound", "Agent not found in Redis", {"agent_id": agent_id, "candidates_tried": list(seen)})
        return None
    except Exception as e:
        logger.log("fetch.agent.redis.exception", "Unexpected error fetching agent", {"agent_id": agent_id, "error": str(e)})
        return None

def fetch_agent_tools(agent_user_id: str) -> Optional[Dict[str, Any]]:
    if not _redis_client:
        logger.log("fetch.tools.no_redis", "Redis client not available", {"agent_user_id": agent_user_id})
        return None
    def _agent_match(row_val: Any, target: Any) -> bool:
        row_str = str(row_val)
        tgt_str = str(target)
        row_core = row_str[5:] if row_str.startswith("agent") else row_str
        tgt_core = tgt_str[5:] if tgt_str.startswith("agent") else tgt_str
        return row_str == tgt_str or row_core == tgt_core
    try:
        table_name = "all-agents-tools"
        ids_key = f"table:{table_name}:ids"
        row_ids = _redis_client.smembers(ids_key) or set()
        row_ids_list: List[str] = []
        row_keys: List[str] = []
        for row_id in row_ids:
            if row_id == "_meta":
                continue
            row_ids_list.append(str(row_id))
            row_keys.append(f"table:{table_name}:row:{row_id}")
        if not row_keys:
            logger.log("fetch.tools.redis.empty", "No tool rows found", {"agent_user_id": agent_user_id})
            return {}
        # Attempt pipelined fetch to avoid per-row RTTs
        fetched_rows: List[Dict[str, Any]] = []
        used_pipeline = False
        try:
            pipe_factory = getattr(_redis_client, "pipeline", None)
            if callable(pipe_factory):
                pipe = pipe_factory()
                for rk in row_keys:
                    pipe.hgetall(rk)
                results = pipe.execute()
                used_pipeline = True
                for rid, res in zip(row_ids_list, results):
                    fetched_rows.append({"row_id": rid, "row": res or {}})
            else:
                raise AttributeError("pipeline not available")
        except Exception as e:
            logger.log("fetch.tools.redis.pipeline_fallback", "Pipeline fetch failed or unavailable, falling back to per-key hgetall", {"error": str(e)})
            fetched_rows = []
            for rid, rk in zip(row_ids_list, row_keys):
                try:
                    row = _redis_client.hgetall(rk) or {}
                    fetched_rows.append({"row_id": rid, "row": row})
                except Exception as e2:
                    logger.log("fetch.tools.row.error", "Failed to hgetall for row", {"row_key": rk, "error": str(e2)})
                    continue
        rows: List[Dict[str, Any]] = []
        for item in fetched_rows:
            row = item.get("row") or {}
            if not row:
                continue
            row_agent_val = row.get("agent_id", "")
            if not _agent_match(row_agent_val, agent_user_id):
                continue
            rows.append({"row_id": item.get("row_id"), "row": row})
        status_order = {"ENABLED": 0, "DISABLED": 1}
        rows.sort(key=lambda r: status_order.get(str(r["row"].get("status", "")).upper(), 99))
        tool_map: Dict[str, Any] = {}
        for item in rows:
            row = item["row"]
            status = str(row.get("status", "") or "").upper()
            if status == "DISABLED":
                continue
            tool_map[item["row_id"]] = {
                "api_url": row.get("api_url", "") or "",
                "api_payload_json": row.get("api_payload_json", "") or "",
                "instructions": row.get("instructions", "") or "",
                "when_run": row.get("when_run", "") or "",
            }
        logger.log("fetch.tools.redis", "Fetched tools from Redis", {"agent_user_id": agent_user_id, "count": len(tool_map), "used_pipeline": used_pipeline})
        return tool_map
    except Exception as e:
        logger.log("fetch.tools.redis.exception", "Error fetching tools from Redis", {"agent_user_id": agent_user_id, "error": str(e)})
        return None

# ---------------------------
# Conversation persistence helpers (all_conversations table)
# ---------------------------
def _parse_json_field(val: Any) -> Any:
    if isinstance(val, str):
        return _safe_json_loads(val)
    return val

def _fetch_conversation_by_phone(phone: str) -> Optional[Dict[str, Any]]:
    if not _redis_client or not phone:
        return None
    try:
        table_name = "all_conversations"
        ids_key = f"table:{table_name}:ids"
        row_ids = _redis_client.smembers(ids_key) or set()
        for row_id in row_ids:
            if row_id == "_meta":
                continue
            row_key = f"table:{table_name}:row:{row_id}"
            try:
                row = _redis_client.hgetall(row_key) or {}
            except Exception as e:
                logger.log("fetch.convo.row.error", "Failed to hgetall for conversation row", {"row_key": row_key, "error": str(e)})
                continue
            recv = row.get("reciever_phone") or row.get("receiver_phone")
            if recv and str(recv) == str(phone):
                row_data = {"id": row_id}
                row_data.update(row)
                for fld in ("conversation_json", "tool_run_logs", "variables"):
                    if fld in row_data:
                        row_data[fld] = _parse_json_field(row_data[fld])
                return row_data
        return None
    except Exception as e:
        logger.log("fetch.convo.exception", "Error fetching conversation by phone", {"phone": phone, "error": str(e)})
        return None

def _fetch_conversation_by_conversation_id(conversation_id: str) -> Optional[Dict[str, Any]]:
    if not _redis_client or not conversation_id:
        return None
    try:
        table_name = "all_conversations"
        ids_key = f"table:{table_name}:ids"
        row_ids = _redis_client.smembers(ids_key) or set()
        for row_id in row_ids:
            if row_id == "_meta":
                continue
            row_key = f"table:{table_name}:row:{row_id}"
            try:
                row = _redis_client.hgetall(row_key) or {}
            except Exception as e:
                logger.log("fetch.convo_by_id.row.error", "Failed to hgetall for conversation row", {"row_key": row_key, "error": str(e)})
                continue
            if not row:
                continue
            cid = row.get("conversation_id")
            if cid and str(cid) == str(conversation_id):
                row_data = {"id": row_id}
                row_data.update(row)
                for fld in ("conversation_json", "tool_run_logs", "variables"):
                    if fld in row_data:
                        row_data[fld] = _parse_json_field(row_data[fld])
                return row_data
        return None
    except Exception as e:
        logger.log("fetch.convo_by_id.exception", "Error fetching conversation by id", {"conversation_id": conversation_id, "error": str(e)})
        return None

def _hset_map(client: UpstashRedis, key: str, mapping: Dict[str, Any]) -> None:
    if not mapping:
        return
    clean_map: Dict[str, str] = {}
    for fld, val in mapping.items():
        if isinstance(val, (dict, list)):
            clean_map[fld] = json.dumps(val)
        elif val is None:
            clean_map[fld] = ""
        elif isinstance(val, bool):
            clean_map[fld] = "1" if val else "0"
        else:
            clean_map[fld] = str(val)
    client.hset(key, values=clean_map)

def _upsert_conversation_row(row_id: Optional[str], data: Dict[str, Any]) -> Optional[str]:
    if not _redis_client:
        logger.log("save.convo.no_redis", "Redis client not available", {})
        return None
    try:
        table_name = "all_conversations"
        if not row_id:
            row_id = str(uuid.uuid4())
        ids_key = f"table:{table_name}:ids"
        row_key = f"table:{table_name}:row:{row_id}"
        to_store: Dict[str, str] = {}
        for k, v in (data or {}).items():
            if v is None:
                continue
            if isinstance(v, (dict, list)):
                try:
                    to_store[k] = json.dumps(v, ensure_ascii=False)
                    continue
                except Exception:
                    pass
            to_store[k] = str(v)
        _redis_client.sadd(ids_key, row_id)
        if to_store:
            _hset_map(_redis_client, row_key, to_store)
        logger.log("save.convo", "Conversation upserted", {"row_id": row_id})
        return row_id
    except Exception as e:
        logger.log("save.convo.error", "Failed to upsert conversation", {"error": str(e)})
        return None

def _upsert_voice_conversation(conversation_id: str, agent_id: str, conversation_json: List[Dict[str, Any]], variables: Optional[Dict[str, Any]] = None, tool_run_logs: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    existing = _fetch_conversation_by_conversation_id(conversation_id)
    row_id = existing.get("id") if isinstance(existing, dict) else None
    data = {
        "conversation_id": conversation_id,
        "agent_id": agent_id,
        "conversation_json": conversation_json,
        "type": "voice-retell",
        "variables": variables or {},
    }
    if tool_run_logs is not None:
        data["tool_run_logs"] = tool_run_logs
    return _upsert_conversation_row(row_id, data)

# ---------------------------
# Utility validators and helpers
# ---------------------------
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_OPTIONAL_KEYS = {"description", "notes", "memo", "comments", "comment"}

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
        t = t.replace("json\n", "", 1).replace("json\r\n", "", 1)
    return t.strip()

def _format_sms_xml(message: str) -> str:
    safe_message = escape(message or "")
    return '<?xml version="1.0" encoding="UTF-8"?>\n<Response>\n  <Message>' + safe_message + "</Message>\n</Response>"

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

def _variables_array_to_dict(variables: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if isinstance(variables, dict):
        for k, v in variables.items():
            if k is None:
                continue
            out[str(k)] = "" if v is None else str(v)
        return out
    if not isinstance(variables, list):
        return out
    for item in variables:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("key")
        if name is None:
            continue
        out[str(name)] = "" if item.get("value") is None else str(item.get("value"))
    return out

def _variables_dict_to_object(variables: Any) -> Dict[str, str]:
    if not isinstance(variables, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in variables.items():
        out[str(k)] = "" if v is None else str(v)
    return out

def _latency_dict_to_csv(lat_map: Dict[str, Any]) -> str:
    """
    Render a latency dictionary into a small CSV string.
    """
    if not isinstance(lat_map, dict):
        return ""
    lines = ["step,ms"]
    for name, val in lat_map.items():
        try:
            ms = int(val)
        except Exception:
            ms = val
        lines.append(f"{name},{ms}")
    return "\n".join(lines)

# ---------------------------
# Static/dynamic prompt builders for prefix caching
# ---------------------------
def _format_tool_schema_for_prompt(tool_id: str, cfg: Dict[str, Any]) -> str:
    """
    Deterministically format tool schema for the static header.
    Avoids variables to keep prefix cache stable.
    """
    api_url = cfg.get("api_url", "") or ""
    payload = cfg.get("api_payload_json", "") or ""
    instructions = cfg.get("instructions", "") or ""
    when_run = cfg.get("when_run", "") or ""
    tool_json = {
        "tool_id": str(tool_id),
        "api_url": api_url,
        "payload_template": payload,
        "instructions": instructions,
        "when_run": when_run,
    }
    return json.dumps(tool_json, ensure_ascii=True, sort_keys=True)

def _build_static_header(merged_config: Dict[str, Any]) -> str:
    """
    Construct the static, cacheable system header. No variables allowed.
    Includes agent prompt, rulebook, and deterministically ordered tool schemas.
    """
    # Static rulebook (no variables/time). Keep fully constant for prefix caching.
    rulebook = (
        "Runtime rulebook for planning and tool calls.\n"
        "You are the runtime planner and reply generator. Before proposing any external action, read the tool's INSTRUCTIONS and AskGuidance tags. For every field marked AskGuidance=SHOULD_BE_ASKED you must either have that exact value provided by the user earlier in the conversation or ask the user a single clear question requesting it. Do not invent or guess values for SHOULD_BE_ASKED fields. If any SHOULD_BE_ASKED field is missing or unclear, output a single question asking for that field and stop. Your output must follow the planner JSON schema and must never call a tool until all SHOULD_BE_ASKED fields are satisfied.\n"
        "INSTRCUTION SEPERATES THE PROPS BASED ON NEED OF ASKING THE PROP'S VALUE OR REFFERENCE FROM THE USER: SHOULD_BE_ASKED CAN_BE_ASKED AND NOT_TO_BE_ASKED, Always ask the Should be Asked FIleds, Never generally Can be asked can be asked from the user, only ask if it is important in the usecase or to compelte the action, NEVER ask the not to be asked fields.\n"
        "If a tool INSTRUCTIONS field for a property is exactly 'EMPTY', then set that property to an empty value with the correct type based on the payload template (e.g., \"\", [], {}, 0, false, etc. based on the value type, set an empty for it.) and do not ask the user for it.\n"
        "When asking, ask exactly one question that requests only the missing information and include the exact field name the system expects.\n"
        "When running a tool, ensure the payload structure matches the tool payload template exactly. If a tool returns a 'needs_input' response, surface that question to the user immediately and stop.\n"
        "Do not claim success unless a tool response confirms success in JSON. Keep replies user-facing and concise only after tools confirm success.\n"
        "Use CURRENT AGENT VARIABLES as the source of truth for user facts. If a relevant variable exists, answer using it and do not ask the user to repeat it.\n"
    )
    tool_lines: List[str] = []
    try:
        for tid in sorted(merged_config.keys(), key=lambda x: str(x)):
            cfg = merged_config.get(tid) or {}
            tool_lines.append(_format_tool_schema_for_prompt(str(tid), cfg))
    except Exception:
        pass
    tools_section = "\n".join(tool_lines)
    static_header = (
        f"GLOBAL_PROMPT:\nYou are a helpful assistant.\n\n"
        f"RUNTIME_RULEBOOK:\n{rulebook}\n\n"
        f"TOOLS_SCHEMAS (sorted):\n{tools_section}"
    )
    return static_header

def _build_dynamic_header(current_vars: Dict[str, Any]) -> str:
    """
    Dynamic header that can change per request without polluting the static prefix.
    """
    try:
        vars_str = json.dumps(current_vars or {}, ensure_ascii=False, sort_keys=True)
    except Exception:
        vars_str = str(current_vars)
    return f"DYNAMIC_AGENT_PROMPT:\n{{agent_prompt}}\n\nCURRENT AGENT VARIABLES:\n{vars_str}"

def _extract_text_from_chunk(chunk: Any) -> str:
    """
    Safely extract text content from a LangChain / OpenAI chunk structure.
    """
    try:
        # LangChain OpenAI chat chunk usually exposes .content as list of AIMessageChunk content parts
        content = getattr(chunk, "content", None)
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, str):
                    parts.append(c)
                else:
                    text_val = getattr(c, "text", None) or getattr(c, "data", None)
                    if isinstance(text_val, str):
                        parts.append(text_val)
            return "".join(parts)
        if isinstance(content, str):
            return content
        # Fallback: try model_dump
        if hasattr(chunk, "model_dump"):
            dumped = chunk.model_dump()
            txt_parts = []
            for item in dumped.get("choices", []):
                delta = item.get("delta", {})
                if "content" in delta and isinstance(delta["content"], list):
                    for c in delta["content"]:
                        if isinstance(c, str):
                            txt_parts.append(c)
                        elif isinstance(c, dict) and "text" in c:
                            txt_parts.append(c["text"])
            return "".join(txt_parts)
    except Exception:
        pass
    return ""

def _is_simple_greeting(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip().lower()
    return bool(re.match(r"^(hi|hello|hey|heyy|hiya|yo|sup)[\\.!\\s]*$", t))

def _query_params_to_dict(qp: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        items = qp.multi_items() if hasattr(qp, "multi_items") else (qp.items() if hasattr(qp, "items") else [])
        for k, v in items:
            if k in out:
                if isinstance(out[k], list):
                    out[k].append(v)
                else:
                    out[k] = [out[k], v]
            else:
                out[k] = v
    except Exception:
        pass
    return out

def _get_case_insensitive(payload: Dict[str, Any], key: str) -> Any:
    if not isinstance(payload, dict):
        return None
    if key in payload:
        return payload.get(key)
    low_key = key.lower()
    for k, v in payload.items():
        if isinstance(k, str) and k.lower() == low_key:
            return v
    return None

def _parse_ask_guidance(instructions_text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not instructions_text:
        return out
    for m in re.finditer(r"'([^']+)'.*?AskGuidance=([A-Z_]+)", instructions_text):
        field_path = m.group(1).strip()
        guidance_raw = m.group(2).strip()
        guidance = "SHOULD_BE_ASKED" if guidance_raw == "SHOULD_BE_ASKED" else "NOT_TO_BE_ASKED"
        if field_path:
            out[field_path] = guidance
    return out

# ---------------------------
# Variable management tool
# ---------------------------
class ManageVariablesArgs(BaseModel):
    model_config = ConfigDict(extra="allow")
    updates: Optional[Dict[str, str]] = None

def manage_variables(state: Optional[dict] = None, updates: Optional[Dict[str, str]] = None, **kwargs: Any) -> Any:
    """
    Use this tool to save, update, or create variables in your internal memory.
    Example: {'user_preference': 'prefers_email'} or updates={'user_preference': 'prefers_email'}
    """
    merged: Dict[str, str] = {}
    if isinstance(updates, dict):
        merged.update(updates)
    for k, v in (kwargs or {}).items():
        merged[str(k)] = "" if v is None else str(v)
    sanitized: Dict[str, str] = {}
    for k, v in merged.items():
        if k is None:
            continue
        sanitized[str(k)] = "" if v is None else str(v)
    # Update process-level runtime state for this request
    runtime_vars = globals().get("_CURRENT_AGENT_VARIABLES", {})
    if not isinstance(runtime_vars, dict):
        runtime_vars = {}
    runtime_vars.update(sanitized)
    globals()["_CURRENT_AGENT_VARIABLES"] = runtime_vars
    # If state is injected (newer LangGraph), update it too
    if isinstance(state, dict):
        current = state.get("variables", {})
        if not isinstance(current, dict):
            current = {}
        current.update(sanitized)
        state["variables"] = current
    return {"variables": sanitized}

MANAGE_VARIABLES_TOOL = StructuredTool.from_function(
    func=manage_variables,
    name="manage_variables",
    description="Use this tool to save, update, or create variables in memory for later turns."
    ,args_schema=ManageVariablesArgs
)

# ---------------------------
# Field Resolution Engine (NEW)
# ---------------------------
# Lightweight evidence extraction regexes
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\d{10}|\d{3}[\s-]\d{3}[\s-]\d{4}|\d{5}[\s-]\d{5})")
_DATE_KEYWORDS = re.compile(r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|tomorrow|today|next\s+\w+)\b", re.I)

def _gather_evidence_from_conversation(conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {"emails": [], "phones": [], "dates": [], "intent_keywords": [], "names": []}
    if not conversation:
        return evidence
    text = " ".join((turn.get("content") or turn.get("message") or "") for turn in (conversation or [])).strip()
    if not text:
        return evidence
    # emails
    for m in re.finditer(r"[\w.+-]+@[\w-]+\.[\w.-]+", text):
        evidence["emails"].append(m.group(0))
    # phones
    for m in _PHONE_RE.finditer(text):
        evidence["phones"].append(m.group(0))
    # dates / date keywords
    for m in _DATE_KEYWORDS.finditer(text):
        evidence["dates"].append(m.group(0))
    # simple intent keywords for appointment types
    for kw in ("cleaning", "checkup", "extraction", "consult", "filling", "root canal", "crown", "book", "appointment", "reschedule", "resched"):
        if re.search(r"\b" + re.escape(kw) + r"\b", text, re.I):
            evidence["intent_keywords"].append(kw)
    # simple name heuristics: "My name is X"
    nm = re.search(r"\b(?:my name is|i am|this is)\s+([A-Z][a-z]{2,20})\b", text)
    if nm:
        evidence["names"].append(nm.group(1))
    return evidence

def _auto_fill_for_field(field: str, payload: Dict[str, Any], template: Dict[str, Any], evidence: Dict[str, Any], agent_prompt: str) -> Optional[Any]:
    return None

def _resolve_fields_and_produce_question(template_obj: Dict[str, Any], ask_map: Dict[str, str], payload: Dict[str, Any], conversation: List[Dict[str, Any]], agent_prompt: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Returns (ok, question_if_any, final_payload).
    ok True means ready to call. If not ok, question is the single question to ask user.
    """
    tpl = template_obj or {}
    evidence = _gather_evidence_from_conversation(conversation or [])
    final = dict(payload or {})
    missing_should: List[str] = []

    for field in tpl.keys():
        cur_val = final.get(field)
        # if non-empty, ok
        if not _is_empty(cur_val):
            continue
        guidance = (ask_map or {}).get(field, None)
        # Respect explicit ask guidance
        if guidance == "NOT_TO_BE_ASKED":
            continue
        # respect ask_map
        if guidance == "SHOULD_BE_ASKED":
            missing_should.append(field)
            continue
        if guidance == "CAN_BE_ASKED":
            # optional, do not block
            continue
        # optional keys skip
        if field in _OPTIONAL_KEYS:
            continue
        # otherwise skip
        continue

    if missing_should:
        chosen = missing_should[0]
        q = f"Could you provide {chosen} so I can complete the booking?"
        return False, q, final

    return True, None, final

def _validate_payload_with_template_and_askmap(payload: Dict[str, Any], template_obj: Any, ask_map: Dict[str, str], conversation: Optional[List[Dict[str, Any]]] = None, agent_prompt: str = "") -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Backwards-compatible wrapper that uses the deterministic resolver.
    """
    if not isinstance(payload, dict):
        return False, "Payload must be a JSON object.", {}
    if not isinstance(template_obj, dict):
        return True, None, payload
    # Ensure template keys exist in payload (allow empty so resolver can decide)
    for k in template_obj.keys():
        if k not in payload:
            payload[k] = payload.get(k, "")
    ok, question, final_payload = _resolve_fields_and_produce_question(template_obj, ask_map or {}, payload, conversation or [], agent_prompt or "")
    return ok, question, final_payload

# ---------------------------
# End Field Resolution Engine
# ---------------------------

def _is_valid_api_url(u: str) -> bool:
    try:
        p = urllib.parse.urlparse(u)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False

# ---------------------------
# Activepieces error helpers
# ---------------------------
_ACTIVE_PIECES_KEY_CACHE: Optional[str] = None
_ACTIVE_PIECES_PROJECT_ID = "m6IFepCya4gsolsay8PoM"

def _fetch_active_pieces_key_from_redis() -> Optional[str]:
    if not _redis_client:
        logger.log("error.fetch_active_pieces_key.redis_unavailable", "Redis client not available for API key lookup", {})
        return None
    try:
        table_name = "API Keys"
        ids_key = f"table:{table_name}:ids"
        row_ids = _redis_client.smembers(ids_key) or set()
        sorted_ids = sorted([rid for rid in row_ids if rid and rid != "_meta"])
        if not sorted_ids:
            logger.log("error.fetch_active_pieces_key.redis_empty", "No API key rows found in Redis", {"table": table_name})
            return None
        first_id = sorted_ids[0]
        row_key = f"table:{table_name}:row:{first_id}"
        row = _redis_client.hgetall(row_key) or {}
        raw_value = row.get("value")
        if isinstance(raw_value, str):
            parsed = _safe_json_loads(raw_value)
            if isinstance(parsed, dict) and parsed.get("value"):
                return str(parsed.get("value"))
        if raw_value:
            return str(raw_value)
        logger.log("error.fetch_active_pieces_key.redis_missing_value", "API key row missing value field", {"row_id": first_id})
        return None
    except Exception as e:
        logger.log("error.fetch_active_pieces_key.redis_error", "Failed to fetch API key from Redis", {"error": str(e)})
        return None

def _extract_flow_id_from_url(api_url: str) -> Optional[str]:
    try:
        parsed = urllib.parse.urlparse(api_url)
        parts = [p for p in parsed.path.split("/") if p]
        if "webhooks" in parts:
            idx = parts.index("webhooks")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        if len(parts) >= 2:
            return parts[-2]
        return parts[-1] if parts else None
    except Exception:
        return None

def _fetch_active_pieces_key() -> Optional[str]:
    global _ACTIVE_PIECES_KEY_CACHE
    if _ACTIVE_PIECES_KEY_CACHE:
        return _ACTIVE_PIECES_KEY_CACHE
    redis_key = _fetch_active_pieces_key_from_redis()
    if redis_key:
        _ACTIVE_PIECES_KEY_CACHE = redis_key
        return _ACTIVE_PIECES_KEY_CACHE
    return None

def _fetch_failed_flow_run(flow_id: str, app_key: str) -> Optional[Dict[str, Any]]:
    base_url = "https://activepieces-production-0d12.up.railway.app/api/v1"
    headers = {"Authorization": app_key}
    list_url = f"{base_url}/flow-runs?flowId={flow_id}&projectId={_ACTIVE_PIECES_PROJECT_ID}&limit=1&status=FAILED"
    try:
        logger.log("internal.request", "Fetching failed flow runs list", {"method": "GET", "url": list_url, "headers": {"Authorization": "(redacted)"}})
        list_resp = requests.get(list_url, headers=headers, timeout=20)
        list_resp_text = None
        try:
            list_resp_text = list_resp.text
        except Exception:
            list_resp_text = None
        logger.log("internal.response", "Failed flow runs list response", {"status_code": list_resp.status_code, "body": (list_resp_text[:1000] + "…") if isinstance(list_resp_text, str) and len(list_resp_text) > 1000 else list_resp_text})
        list_data = list_resp.json() if list_resp.ok else None
        if not isinstance(list_data, dict):
            logger.log("error.flow_runs.parse", "Failed to parse failed flow runs list", {"status": list_resp.status_code if list_resp else None})
            return None
        runs = list_data.get("data") if isinstance(list_data.get("data"), list) else []
        if not runs:
            logger.log("error.flow_runs.empty", "No failed flow runs found", {"flow_id": flow_id})
            return None
        run_id = runs[0].get("id")
        if not run_id:
            logger.log("error.flow_runs.no_id", "Failed run missing id", {"flow_id": flow_id})
            return None
        detail_url = f"{base_url}/flow-runs/{run_id}"
        logger.log("internal.request", "Fetching failed flow run detail", {"method": "GET", "url": detail_url, "headers": {"Authorization": "(redacted)"}})
        detail_resp = requests.get(detail_url, headers=headers, timeout=20)
        detail_resp_text = None
        try:
            detail_resp_text = detail_resp.text
        except Exception:
            detail_resp_text = None
        logger.log("internal.response", "Failed flow run detail response", {"status_code": detail_resp.status_code, "body": (detail_resp_text[:1000] + "…") if isinstance(detail_resp_text, str) and len(detail_resp_text) > 1000 else detail_resp_text})
        detail_json = detail_resp.json() if detail_resp.ok else None
        if not isinstance(detail_json, dict):
            logger.log("error.flow_run.detail_parse", "Failed to parse flow run detail", {"status": detail_resp.status_code if detail_resp else None, "run_id": run_id})
            return None
        return detail_json
    except Exception as e:
        logger.log("error.flow_runs.fetch", "Failed to fetch failed flow run details", {"error": str(e), "flow_id": flow_id})
        return None

def _extract_error_details_from_run(run_detail: Dict[str, Any]) -> Tuple[Optional[str], Optional[Any]]:
    error_msg = None
    trigger_raw_body = None
    steps = run_detail.get("steps") if isinstance(run_detail, dict) else {}
    if isinstance(steps, dict):
        failed_step_name = None
        failed_step = run_detail.get("failedStep")
        if isinstance(failed_step, dict):
            failed_step_name = failed_step.get("name")
        step_key = failed_step_name or "step_1"
        step_info = steps.get(step_key) if isinstance(steps.get(step_key), dict) else steps.get("step_1")
        if isinstance(step_info, dict):
            error_msg = step_info.get("errorMessage")
        trigger_info = steps.get("trigger")
        if isinstance(trigger_info, dict):
            output = trigger_info.get("output")
            if isinstance(output, dict):
                trigger_raw_body = output.get("rawBody")
    return error_msg, trigger_raw_body

def _build_activepieces_error_message(tool_display_name: str, api_url: str) -> Tuple[str, Dict[str, Any]]:
    flow_id = _extract_flow_id_from_url(api_url)
    if not flow_id:
        msg = f"{tool_display_name} HAD AN ISSUE, DETAILS: Unable to parse flow id from API URL."
        return msg, {"error": "flow_id_unavailable"}
    app_key = _fetch_active_pieces_key()
    if not app_key:
        msg = f"{tool_display_name} HAD AN ISSUE, DETAILS: Unable to fetch Activepieces API key."
        return msg, {"error": "app_key_unavailable", "flow_id": flow_id}
    run_detail = _fetch_failed_flow_run(flow_id, app_key)
    if not isinstance(run_detail, dict):
        msg = f"{tool_display_name} HAD AN ISSUE, DETAILS: Unable to fetch flow run details."
        return msg, {"error": "run_detail_unavailable", "flow_id": flow_id}
    error_msg, trigger_raw_body = _extract_error_details_from_run(run_detail)
    err_text = error_msg or "Unknown error"
    raw_body_text = trigger_raw_body if isinstance(trigger_raw_body, str) else json.dumps(trigger_raw_body, ensure_ascii=False) if trigger_raw_body is not None else "Unavailable"
    message = f"{tool_display_name} HAD AN ISSUE, DETAILS: {err_text} ; trriger raw body: {raw_body_text}"
    detail_payload = {"flow_id": flow_id, "error_message": error_msg, "trigger_raw_body": trigger_raw_body}
    return message, detail_payload

# ---------------------------
# Tool factory
# ---------------------------
def create_universal_tools(config: Dict[str, Any]) -> List[StructuredTool]:
    generated_tools: List[StructuredTool] = []
    for tool_id, cfg in (config or {}).items():
        api_url = str(cfg.get("api_url", "") or "").strip()
        instructions = str(cfg.get("instructions", "") or "")
        when_run = str(cfg.get("when_run", "") or "")
        raw_payload_template = urllib.parse.unquote(str(cfg.get("api_payload_json", "") or ""))
        tpl_obj = _safe_json_loads(raw_payload_template)
        ask_map = _parse_ask_guidance(instructions)
        DynamicArgsModel = None
        if isinstance(tpl_obj, dict) and tpl_obj:
            try:
                fields = {k: (Any, Field(description=f"Value for {k}")) for k in tpl_obj.keys()}
                DynamicArgsModel = create_model(f"Args_{tool_id}", **fields)
            except Exception:
                DynamicArgsModel = None
        def _make_api_call_factory(_tool_id: str, _api_url: str, _tpl_obj: Any, _ask_map: Dict[str, str]):
            def make_api_call(state: Annotated[Optional[dict], InjectedState] = None, **kwargs) -> str:
                event_id = str(uuid.uuid4())
                tool_display_name = f"sync_data_tool_{_tool_id}"
                payload = dict(kwargs or {})
                logger.log("tool.call", f"api_tool_{_tool_id} triggered", {"event_id": event_id, "api_url": _api_url, "payload": payload})
                # Determine conversation and agent_prompt context to pass to resolver.
                # Prefer explicit kwargs override, otherwise fallback to runtime globals set before agent.invoke.
                conversation_for_context = None
                agent_prompt_for_context = None
                if "_conversation" in payload:
                    conversation_for_context = payload.pop("_conversation")
                if "_agent_prompt" in payload:
                    agent_prompt_for_context = payload.pop("_agent_prompt")
                if conversation_for_context is None:
                    conversation_for_context = globals().get("_CURRENT_RUNTIME_CONVERSATION", [])
                if agent_prompt_for_context is None:
                    agent_prompt_for_context = globals().get("_CURRENT_AGENT_PROMPT", "")
                current_vars: Dict[str, Any] = {}
                try:
                    if isinstance(state, dict):
                        current_vars = state.get("variables", {}) or {}
                except Exception:
                    current_vars = {}
                if not current_vars:
                    runtime_vars = globals().get("_CURRENT_AGENT_VARIABLES", {})
                    if isinstance(runtime_vars, dict):
                        current_vars = runtime_vars
                # Validate using the new resolver (passes conversation and agent prompt)
                ok, question, payload2 = _validate_payload_with_template_and_askmap(payload, _tpl_obj, _ask_map, conversation_for_context, agent_prompt_for_context)
                if not ok:
                    tool_result = {"ok": False, "status_code": None, "response": None, "event_id": event_id, "needs_input": True, "question": question}
                    logger.log("tool.validation", f"api_tool_{_tool_id} needs user input", tool_result)
                    return tool_result
                if not _is_valid_api_url(_api_url):
                    error_data = {"ok": False, "status_code": None, "response": None, "event_id": event_id, "error": "Invalid api_url"}
                    logger.log("tool.error", f"api_tool_{_tool_id} invalid api_url", error_data)
                    return error_data
                def _attach_error_details(base_result: Dict[str, Any]) -> Dict[str, Any]:
                    # Ensure we always mark this as an error and swap response with enriched context.
                    merged = dict(base_result)
                    merged["ok"] = False
                    merged["error"] = True
                    message, detail_payload = _build_activepieces_error_message(tool_display_name, _api_url)
                    merged["response"] = message
                    merged["error_details"] = detail_payload
                    return merged
                try:
                    # Do not inject context variables into tool payloads.
                    resp = requests.post(_api_url, json=payload2, timeout=20)
                    try:
                        response_data = resp.json()
                    except Exception:
                        response_data = resp.text
                    tool_result = {"ok": bool(resp.ok), "status_code": resp.status_code, "response": response_data, "event_id": event_id}
                    if resp.status_code == 500 or tool_result.get("ok") is False:
                        tool_result = _attach_error_details(tool_result)
                    logger.log("tool.response", f"api_tool_{_tool_id} result", tool_result)
                    return tool_result
                except Exception as e:
                    error_data = {"ok": False, "status_code": None, "response": None, "event_id": event_id, "error": str(e)}
                    error_data = _attach_error_details(error_data)
                    logger.log("tool.error", f"api_tool_{_tool_id} failed", error_data)
                    return error_data
            return make_api_call
        make_api_call = _make_api_call_factory(str(tool_id), api_url, tpl_obj, ask_map)
        description = (f"WHEN_RUN: {when_run}\nINSTRUCTIONS: {instructions}\nPAYLOAD_TEMPLATE: {raw_payload_template}\nDo not invent missing details. Ask if unsure.")
        new_tool = StructuredTool.from_function(func=make_api_call, name=f"sync_data_tool_{tool_id}", description=description, args_schema=DynamicArgsModel)
        generated_tools.append(new_tool)
    return generated_tools

# ---------------------------
# Convert conversation to messages
# ---------------------------
def _to_messages(conversation_history: List[Dict[str, Any]], user_message: str) -> List[Any]:
    msgs: List[Any] = []
    for turn in (conversation_history or []):
        role = str(turn.get("role", "") or "").lower().strip()
        content = turn.get("content") if "content" in turn else turn.get("message")
        if content is None:
            content = ""
        if role == "user":
            msgs.append(HumanMessage(content=str(content)))
        elif role == "assistant":
            msgs.append(AIMessage(content=str(content)))
        else:
            msgs.append(HumanMessage(content=str(content)))
    if user_message is not None:
        msgs.append(HumanMessage(content=str(user_message)))
    return msgs

# ---------------------------
# Core run logic
# ---------------------------
def run_agent(agent_id: str, conversation_history: List[Dict[str, Any]], message: str, variables: Optional[Any] = None, stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    # Synchronous wrapper for compatibility; executes the async core
    return asyncio.run(run_agent_async(agent_id, conversation_history, message, variables, stream_callback))

async def run_agent_async(agent_id: str, conversation_history: List[Dict[str, Any]], message: str, variables: Optional[Any] = None, stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    logger.clear()
    logger.log("run.start", "Agent started", {"input_agent_id": agent_id})
    overall_start = time.perf_counter()
    latencies: Dict[str, int] = {}

    def _mark_latency(name: str, start_time: float):
        try:
            latencies[name] = max(int((time.perf_counter() - start_time) * 1000), 0)
        except Exception:
            latencies[name] = 0

    def _finalize_output(base: Dict[str, Any]) -> Dict[str, Any]:
        total_ms = max(int((time.perf_counter() - overall_start) * 1000), 0)
        latencies["run_agent_total_ms"] = total_ms
        latencies["combined_latency_ms"] = total_ms
        base["latency_ms"] = dict(latencies)
        base["combined_latency_ms"] = total_ms
        base["latency_csv"] = _latency_dict_to_csv(base["latency_ms"])
        globals()["_CURRENT_STREAM_CALLBACK"] = None
        return base

    initial_vars = _variables_array_to_dict(variables)
    globals()["_CURRENT_AGENT_VARIABLES"] = dict(initial_vars)
    globals()["_CURRENT_STREAM_CALLBACK"] = stream_callback
    t_agent_fetch = time.perf_counter()
    agent_resp = fetch_agent_details(agent_id)
    _mark_latency("fetch_agent_details_ms", t_agent_fetch)
    if not agent_resp:
        logger.log("run.error", "Agent details fetch returned None", {"agent_id": agent_id})
        return _finalize_output({"reply": "Error: Failed to fetch agent details.", "logs": logger.to_list(), "variables": _variables_dict_to_object(initial_vars)})
    api_key_to_use = None
    for key_name in ("api_key", "openai_api_key", "openai_key", "key"):
        if agent_resp.get(key_name):
            api_key_to_use = agent_resp.get(key_name)
            break
    if not api_key_to_use:
        api_key_to_use = os.getenv("OPENAI_API_KEY")
    if not api_key_to_use:
        logger.log("run.error", "OpenAI API key missing", {})
        return _finalize_output({"reply": "Error: Missing OpenAI API key.", "logs": logger.to_list(), "variables": _variables_dict_to_object(initial_vars)})
    agent_prompt = str(agent_resp.get("prompt", "You are a helpful assistant.") or "").strip()

    t_tools = time.perf_counter()
    fetched_tools = fetch_agent_tools(str(agent_id))
    _mark_latency("fetch_agent_tools_ms", t_tools)
    merged_config = dict(DYNAMIC_CONFIG)
    if isinstance(fetched_tools, dict):
        for tid, tcfg in fetched_tools.items():
            merged_config[str(tid)] = {"api_url": tcfg.get("api_url", ""), "api_payload_json": tcfg.get("api_payload_json", ""), "instructions": tcfg.get("instructions", ""), "when_run": tcfg.get("when_run", "")}
    logger.log("run.config", "Merged dynamic config", {"tool_count": len(merged_config)})
    t_build_agent = time.perf_counter()
    # Deterministic tool ordering for caching
    sorted_config_items = sorted(merged_config.items(), key=lambda kv: str(kv[0]))
    ordered_config: Dict[str, Any] = {k: v for k, v in sorted_config_items}
    static_header = _build_static_header(ordered_config)

    def _state_modifier_fn(state: AgentState) -> List[Any]:
        current_vars: Dict[str, Any] = {}
        try:
            if isinstance(state, dict):
                current_vars = state.get("variables", {}) or {}
        except Exception:
            current_vars = {}
        if not current_vars:
            runtime_vars = globals().get("_CURRENT_AGENT_VARIABLES", {})
            if isinstance(runtime_vars, dict):
                current_vars = runtime_vars
        dynamic_header = _build_dynamic_header(current_vars).replace("{agent_prompt}", agent_prompt)
        system_msgs = [
            SystemMessage(content=static_header),
            SystemMessage(content=dynamic_header),
        ]
        messages: List[Any] = []
        if isinstance(state, dict):
            messages = state.get("messages", []) or []
        # Remove any prior system messages to avoid duplication
        filtered = [m for m in messages if not isinstance(m, SystemMessage)]
        return system_msgs + filtered

    tools = [MANAGE_VARIABLES_TOOL] + create_universal_tools(ordered_config)
    llm = ChatOpenAI(
        api_key=api_key_to_use,
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=100,
        top_p=0.9,
        n=1,
        streaming=True,
    )
    # Build agent with in-memory checkpointer to enable streaming
    checkpointer = MemorySaver()
    agent = create_react_agent(
        llm,
        tools,
        state_modifier=_state_modifier_fn,
        state_schema=AgentState,
        checkpointer=checkpointer,
    )
    _mark_latency("build_agent_ms", t_build_agent)
    msgs = _to_messages(conversation_history, message)

    # Inject runtime context globals so tool calls can access conversation and agent prompt deterministically.
    # The resolver prefers explicit _conversation and _agent_prompt kwargs, otherwise falls back to these globals.
    globals()["_CURRENT_RUNTIME_CONVERSATION"] = (conversation_history or []) + [{"role": "user", "content": message}]
    globals()["_CURRENT_AGENT_PROMPT"] = agent_prompt

    # Fast-response short-circuit for simple greetings
    if _is_simple_greeting(message) and not conversation_history:
        fast_llm = ChatOpenAI(
            api_key=api_key_to_use,
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=50,
            top_p=0.9,
            n=1,
            streaming=True,
        )
        fast_chunks: List[str] = []

        async def _fast_stream():
            fast_msgs = [SystemMessage(content=static_header), SystemMessage(content=_build_dynamic_header(initial_vars).replace("{agent_prompt}", agent_prompt))] + msgs
            async for chunk in fast_llm.astream(fast_msgs):
                text_piece = _extract_text_from_chunk(chunk)
                if text_piece:
                    fast_chunks.append(text_piece)
                    if callable(stream_callback):
                        try:
                            stream_callback(text_piece)
                        except Exception:
                            pass
            return "".join(fast_chunks)

        try:
            fast_reply = await _fast_stream()
            reply_text = fast_reply or "Hello!"
        except Exception as e:
            logger.log("run.fast_response.error", "Fast response failed, falling back to main agent", {"error": str(e)})
            reply_text = None
        if reply_text:
            _mark_latency("agent_invoke_ms", t_build_agent)  # approximate for fast path
            return _finalize_output({"reply": reply_text, "logs": logger.to_list(), "variables": _variables_dict_to_object(initial_vars)})
    invoke_start = time.perf_counter()
    state = None
    partial_reply_chunks: List[str] = []

    async def _run_stream():
        local_state = None
        async for event in agent.astream_events(
            {"messages": msgs, "variables": initial_vars, "is_last_step": False},
            stream_mode="values",
            version="v2",
            config={"recursion_limit": 50, "configurable": {"thread_id": str(uuid.uuid4())}},
        ):
            etype = event.get("type", "")
            data = event.get("data", {}) or {}
            if etype == "on_chat_model_stream":
                chunk = data.get("chunk")
                text_piece = _extract_text_from_chunk(chunk)
                if text_piece:
                    partial_reply_chunks.append(text_piece)
                    stream_cb = globals().get("_CURRENT_STREAM_CALLBACK")
                    if callable(stream_cb):
                        try:
                            stream_cb(text_piece)
                        except Exception:
                            pass
            if etype in {"on_chain_end", "on_graph_end", "on_llm_end"}:
                local_state = data.get("output") or data.get("state") or data
            if data.get("messages"):
                local_state = data
        return local_state

    try:
        # Log before model call
        logger.log("agent.invoke.start", "Invoking agent (astream_events)", {"message_count": len(msgs)})
        state = await _run_stream()
        logger.log("agent.invoke.end", "Agent finished stream")
    except GraphRecursionError as ge:
        _mark_latency("agent_invoke_ms", invoke_start)
        # Root-level safeguard: if the graph keeps looping, fall back to a single-shot LLM response
        logger.log("run.error", "Graph recursion limit hit, falling back to direct reply", {"error": str(ge)})
        fallback_prompt = static_header
        try:
            fallback_msgs = [SystemMessage(content=static_header), SystemMessage(content=_build_dynamic_header(initial_vars).replace("{agent_prompt}", agent_prompt))] + msgs
            fallback_start = time.perf_counter()
            fallback_resp = llm.invoke(fallback_msgs)
            _mark_latency("fallback_llm_ms", fallback_start)
            reply_text = fallback_resp.content if hasattr(fallback_resp, "content") else str(fallback_resp)
        except Exception as le:
            logger.log("run.error", "Fallback LLM failed", {"error": str(le)})
            reply_text = f"Error: {str(ge)}"
        return _finalize_output({"reply": reply_text, "logs": logger.to_list(), "variables": _variables_dict_to_object(initial_vars)})
    except Exception as e:
        _mark_latency("agent_invoke_ms", invoke_start)
        logger.log("run.error", "Agent execution exception", {"error": str(e), "traceback": traceback.format_exc()})
        return _finalize_output({"reply": f"Error: {str(e)}", "logs": logger.to_list(), "variables": _variables_dict_to_object(initial_vars)})
    _mark_latency("agent_invoke_ms", invoke_start)
    if state is None:
        # As a fallback, try a single invoke to get state
        try:
            state = agent.invoke({"messages": msgs, "variables": initial_vars, "is_last_step": False}, config={"recursion_limit": 50})
        except Exception:
            state = None
    if state is None:
        reply_text = "".join(partial_reply_chunks) or "Error: No response generated."
        return _finalize_output({"reply": reply_text, "logs": logger.to_list(), "variables": _variables_dict_to_object(initial_vars)})
    # Extract last assistant message
    reply_text = ""
    try:
        out_msgs = state.get("messages", []) if isinstance(state, dict) else []
        last_ai = None
        for m in reversed(out_msgs):
            if isinstance(m, AIMessage):
                last_ai = m
                break
        if last_ai and isinstance(last_ai.content, str):
            reply_text = last_ai.content.strip()
        elif last_ai:
            reply_text = str(last_ai.content)
        else:
            reply_text = "".join(partial_reply_chunks) or "Done."
    except Exception:
        reply_text = "".join(partial_reply_chunks) or "Done."
    final_variables_dict: Dict[str, Any] = dict(initial_vars)
    try:
        if isinstance(state, dict) and isinstance(state.get("variables"), dict):
            final_variables_dict.update(state.get("variables", {}) or {})
    except Exception:
        pass
    runtime_vars = globals().get("_CURRENT_AGENT_VARIABLES", {})
    if isinstance(runtime_vars, dict):
        final_variables_dict.update(runtime_vars)
    # Detect if any tool result asked for more input and surface question
    try:
        out_msgs = state.get("messages", []) if isinstance(state, dict) else []
        for m in out_msgs:
            if isinstance(m, ToolMessage):
                content = m.content
                if isinstance(content, str):
                    parsed = _safe_json_loads(content)
                    if isinstance(parsed, dict) and parsed.get("needs_input"):
                        q = parsed.get("question") or parsed.get("message") or parsed.get("error") or "Could you share the missing details needed to proceed?"
                        logger.log("tool.needs_input", "Tool requested more input", {"question": q, "tool_message": parsed})
                        return _finalize_output({"reply": str(q), "logs": logger.to_list(), "variables": _variables_dict_to_object(final_variables_dict)})
    except Exception:
        pass
    # Compact trace for logs
    try:
        out_msgs = state.get("messages", []) if isinstance(state, dict) else []
        compact_trace: List[Dict[str, Any]] = []
        for m in out_msgs[-40:]:
            if isinstance(m, HumanMessage):
                compact_trace.append({"role": "user", "content": (m.content[:500] + "…") if len(m.content) > 500 else m.content})
            elif isinstance(m, ToolMessage):
                tool_content: Any = m.content
                if isinstance(tool_content, str):
                    parsed = _safe_json_loads(tool_content)
                    if isinstance(parsed, dict):
                        response_val = parsed.get("response")
                        if isinstance(response_val, (dict, list)):
                            parsed["response"] = json.dumps(response_val, ensure_ascii=False)
                        tool_content = parsed
                    if isinstance(tool_content, str) and len(tool_content) > 700:
                        tool_content = tool_content[:700] + "…"
                compact_trace.append({"role": "tool", "content": tool_content})
            elif isinstance(m, AIMessage):
                tc = getattr(m, "tool_calls", None)
                compact_trace.append({"role": "assistant", "content": (m.content[:500] + "…") if isinstance(m.content, str) and len(m.content) > 500 else m.content, "tool_calls": tc})
        logger.log("run.trace", "Final message trace (compact)", {"messages": compact_trace})
    except Exception:
        pass
    return _finalize_output({"reply": reply_text, "logs": logger.to_list(), "variables": _variables_dict_to_object(final_variables_dict)})

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI()
# In dev allow all origins. In production set explicit origins list.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.post("/run-agent")
async def run_endpoint(request: Request):
    request_start = time.perf_counter()
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"reply": "Error: Invalid JSON request body", "logs": logger.to_list()})
    query_params = _query_params_to_dict(request.query_params)
    payload = body.get("input") if isinstance(body.get("input"), dict) else body
    agent_id = payload.get("agent_id") or payload.get("agentId") or payload.get("id") or query_params.get("agent_id") or query_params.get("agentId") or query_params.get("id")
    is_demo_sms = isinstance(payload, dict) and (_get_case_insensitive(payload, "Body") is not None and _get_case_insensitive(payload, "From") is not None)
    conversation = payload.get("conversation", []) if not is_demo_sms else []
    variables_payload = payload.get("variables", [])
    message = payload.get("message", "")
    existing_convo_row = None
    is_first_demo_sms = False
    reciever_phone = None
    if is_demo_sms:
        reciever_phone = str(_get_case_insensitive(payload, "From") or "")
        message = str(_get_case_insensitive(payload, "Body") or "")
        existing_convo_row = _fetch_conversation_by_phone(reciever_phone)
        if existing_convo_row:
            if not agent_id:
                agent_id = existing_convo_row.get("agent_id")
            convo_json = existing_convo_row.get("conversation_json") or []
            if isinstance(convo_json, list):
                conversation = convo_json
            stored_vars = existing_convo_row.get("variables") or {}
            if isinstance(stored_vars, dict):
                variables_payload = stored_vars
        else:
            is_first_demo_sms = True
    if not agent_id:
        elapsed_ms = max(int((time.perf_counter() - request_start) * 1000), 0)
        latency_ms = {"http_handler_total_ms": elapsed_ms, "combined_latency_ms": elapsed_ms}
        return JSONResponse({
            "reply": "Error: Missing agent_id in request",
            "logs": logger.to_list(),
            "latency_ms": latency_ms,
            "combined_latency_ms": elapsed_ms,
            "latency_report_csv": _latency_dict_to_csv(latency_ms)
        })
    if callable(run_agent_async):
        res = await run_agent_async(str(agent_id), conversation, str(message), variables_payload, None)
    else:
        res = await asyncio.to_thread(run_agent, str(agent_id), conversation, str(message), variables_payload, None)
    if is_demo_sms and is_first_demo_sms:
        reply_text = res.get("reply", "")
        if reply_text and not reply_text.startswith("Demo by SaaS: "):
            res["reply"] = f"Demo by SaaS: {reply_text}"
    if is_demo_sms and reciever_phone:
        prev_convo = conversation if isinstance(conversation, list) else []
        new_convo = list(prev_convo)
        new_convo.append({"role": "user", "content": message})
        new_convo.append({"role": "assistant", "content": res.get("reply", "")})
        assistant_index = len(new_convo)
        prior_tool_logs = []
        if existing_convo_row and isinstance(existing_convo_row.get("tool_run_logs"), list):
            prior_tool_logs = existing_convo_row.get("tool_run_logs")
        tool_events = []
        for ev in res.get("logs", []):
            if isinstance(ev, dict) and str(ev.get("type", "")).startswith("tool"):
                ev_copy = dict(ev)
                ev_copy["assistant_index"] = assistant_index
                tool_events.append(ev_copy)
        updated_tool_logs = prior_tool_logs + tool_events
        updated_variables = res.get("variables", {})
        if not isinstance(updated_variables, dict):
            updated_variables = {}
        row_id = existing_convo_row.get("id") if existing_convo_row else None
        _upsert_conversation_row(row_id, {
            "agent_id": agent_id,
            "conversation_json": new_convo,
            "reciever_phone": reciever_phone,
            "tool_run_logs": updated_tool_logs,
            "type": "demo-sms",
            "variables": updated_variables,
        })
    handler_latency_ms = max(int((time.perf_counter() - request_start) * 1000), 0)
    latency_ms_map: Dict[str, Any] = {}
    if isinstance(res.get("latency_ms"), dict):
        latency_ms_map.update(res.get("latency_ms"))
    if "combined_latency_ms" in latency_ms_map:
        latency_ms_map["run_agent_combined_latency_ms"] = latency_ms_map.get("combined_latency_ms")
    latency_ms_map["http_handler_total_ms"] = handler_latency_ms
    latency_ms_map["combined_latency_ms"] = handler_latency_ms
    res["latency_ms"] = latency_ms_map
    res["combined_latency_ms"] = handler_latency_ms
    res["latency_report_csv"] = _latency_dict_to_csv(latency_ms_map)
    if is_demo_sms:
        return Response(content=_format_sms_xml(res.get("reply", "")), media_type="application/xml")
    return JSONResponse(res)

# Support inline execution when running inside CI or sandbox that injects 'inputs'
if "inputs" in globals():
    request_start = time.perf_counter()
    data = globals().get("inputs", {}) or {}
    q = globals().get("query", {}) or {}
    payload = data.get("input") if isinstance(data.get("input"), dict) else data
    agent_id = payload.get("agent_id") or payload.get("agentId") or payload.get("id") or q.get("agent_id") or q.get("agentId") or q.get("id")
    is_demo_sms = isinstance(payload, dict) and (_get_case_insensitive(payload, "Body") is not None and _get_case_insensitive(payload, "From") is not None)
    conversation = payload.get("conversation", []) if not is_demo_sms else []
    variables_payload = payload.get("variables", [])
    message = payload.get("message", "")
    existing_convo_row = None
    is_first_demo_sms = False
    reciever_phone = None
    if is_demo_sms:
        reciever_phone = str(_get_case_insensitive(payload, "From") or "")
        message = str(_get_case_insensitive(payload, "Body") or "")
        existing_convo_row = _fetch_conversation_by_phone(reciever_phone)
        if existing_convo_row:
            if not agent_id:
                agent_id = existing_convo_row.get("agent_id")
            convo_json = existing_convo_row.get("conversation_json") or []
            if isinstance(convo_json, list):
                conversation = convo_json
            stored_vars = existing_convo_row.get("variables") or {}
            if isinstance(stored_vars, dict):
                variables_payload = stored_vars
        else:
            is_first_demo_sms = True
    _out = run_agent(str(agent_id), conversation, str(message), variables_payload)
    if is_demo_sms and is_first_demo_sms:
        reply_text = _out.get("reply", "")
        if reply_text and not reply_text.startswith("Demo by SaaS: "):
            _out["reply"] = f"Demo by SaaS: {reply_text}"
    if is_demo_sms and reciever_phone:
        prev_convo = conversation if isinstance(conversation, list) else []
        new_convo = list(prev_convo)
        new_convo.append({"role": "user", "content": message})
        new_convo.append({"role": "assistant", "content": _out.get("reply", "")})
        assistant_index = len(new_convo)
        prior_tool_logs = []
        if existing_convo_row and isinstance(existing_convo_row.get("tool_run_logs"), list):
            prior_tool_logs = existing_convo_row.get("tool_run_logs")
        tool_events = []
        for ev in _out.get("logs", []):
            if isinstance(ev, dict) and str(ev.get("type", "")).startswith("tool"):
                ev_copy = dict(ev)
                ev_copy["assistant_index"] = assistant_index
                tool_events.append(ev_copy)
        updated_tool_logs = prior_tool_logs + tool_events
        updated_variables = _out.get("variables", {})
        if not isinstance(updated_variables, dict):
            updated_variables = {}
        row_id = existing_convo_row.get("id") if existing_convo_row else None
        _upsert_conversation_row(row_id, {
            "agent_id": agent_id,
            "conversation_json": new_convo,
            "reciever_phone": reciever_phone,
            "tool_run_logs": updated_tool_logs,
            "type": "demo-sms",
            "variables": updated_variables,
        })
    handler_latency_ms = max(int((time.perf_counter() - request_start) * 1000), 0)
    latency_ms_map: Dict[str, Any] = {}
    if isinstance(_out.get("latency_ms"), dict):
        latency_ms_map.update(_out.get("latency_ms"))
    if "combined_latency_ms" in latency_ms_map:
        latency_ms_map["run_agent_combined_latency_ms"] = latency_ms_map.get("combined_latency_ms")
    latency_ms_map["http_handler_total_ms"] = handler_latency_ms
    latency_ms_map["combined_latency_ms"] = handler_latency_ms
    _out["latency_ms"] = latency_ms_map
    _out["combined_latency_ms"] = handler_latency_ms
    _out["latency_report_csv"] = _latency_dict_to_csv(latency_ms_map)
    try:
        _out["debug_inputs"] = {
            "inputs_keys": list(data.keys()) if isinstance(data, dict) else [],
            "payload_keys": list(payload.keys()) if isinstance(payload, dict) else [],
            "from": _get_case_insensitive(payload, "From"),
            "body": _get_case_insensitive(payload, "Body"),
        }
    except Exception:
        pass
    if is_demo_sms:
        globals()["result"] = _format_sms_xml(_out.get("reply", ""))
    else:
        globals()["result"] = _out
