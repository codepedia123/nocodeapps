# main.py - LangGraph create_react_agent runtime with dynamic tools (Redis fetched), reply + logs
import os
import json
import uuid
import traceback
import urllib.parse
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Annotated
from operator import ior
import requests

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from upstash_redis import Redis as UpstashRedis
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.graph import MessagesState

from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI

from pydantic import create_model, Field
from langgraph.errors import GraphRecursionError

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
            row_agent_val = row.get("agent_id", "")
            if not _agent_match(row_agent_val, agent_user_id):
                continue
            tool_map[str(row_id)] = {
                "api_url": row.get("api_url", "") or "",
                "api_payload_json": row.get("api_payload_json", "") or "",
                "instructions": row.get("instructions", "") or "",
                "when_run": row.get("when_run", "") or "",
            }
        logger.log("fetch.tools.redis", "Fetched tools from Redis", {"agent_user_id": agent_user_id, "count": len(tool_map)})
        return tool_map
    except Exception as e:
        logger.log("fetch.tools.redis.exception", "Error fetching tools from Redis", {"agent_user_id": agent_user_id, "error": str(e)})
        return None

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

def _variables_dict_to_array(variables: Any) -> List[Dict[str, str]]:
    if not isinstance(variables, dict):
        return []
    arr: List[Dict[str, str]] = []
    for k, v in variables.items():
        arr.append({"name": str(k), "value": "" if v is None else str(v)})
    return arr

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
def manage_variables(updates: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """
    Use this tool to save, update, or create variables in your internal memory.
    Example: {'user_preference': 'prefers_email'}
    """
    if not isinstance(updates, dict):
        updates = {}
    sanitized: Dict[str, str] = {}
    for k, v in updates.items():
        if k is None:
            continue
        sanitized[str(k)] = "" if v is None else str(v)
    # Returning this shape allows LangGraph to merge into the 'variables' state
    return {"variables": sanitized}

MANAGE_VARIABLES_TOOL = StructuredTool.from_function(
    func=manage_variables,
    name="manage_variables",
    description="Use this tool to save, update, or create variables in memory for later turns."
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
    # end_date_time -> default +30 minutes if start provided and isoparseable
    if field in ("end_date_time", "endTime", "end") and _is_empty(payload.get(field)):
        start = payload.get("start_date_time") or payload.get("start") or payload.get("dtstart")
        if isinstance(start, str) and start:
            try:
                dt = datetime.fromisoformat(start)
                return (dt + timedelta(minutes=30)).isoformat()
            except Exception:
                return None
    # title generation: use intent keywords to create a sensible title
    if field in ("title", "summary") and (_is_empty(payload.get(field))):
        if evidence.get("intent_keywords"):
            kw = evidence["intent_keywords"][0]
            return f"{kw.capitalize()} Appointment"
        if re.search(r"\bdental\b|\bclinic\b", agent_prompt, re.I) and evidence.get("intent_keywords"):
            return "Clinic Appointment"
        return "Appointment"
    # attendees: avoid inventing emails
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

    preferred_not_ask = {"title", "summary", "id", "start_date_time", "end_date_time", "dtstart", "dtend"}

    booking_mode = bool(re.search(r"\b(book|appointment|schedule|resched|reschedule)\b", agent_prompt, re.I))

    for field in tpl.keys():
        cur_val = final.get(field)
        # if non-empty, ok
        if not _is_empty(cur_val):
            continue
        # try deterministic autofill
        auto = _auto_fill_for_field(field, final, tpl, evidence, agent_prompt)
        if auto is not None:
            final[field] = auto
            logger.log("autofill", f"Auto-filled field {field}", {"value": auto})
            continue
        guidance = (ask_map or {}).get(field, None)
        # system preference for not asking certain fields
        if field in preferred_not_ask:
            if guidance == "SHOULD_BE_ASKED":
                pass
            else:
                continue
        # booking mode: treat emails/attendees as required
        if booking_mode and _looks_like_email_field(field):
            missing_should.append(field)
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
        # if field looks like email/attendee, ask
        if _looks_like_email_field(field):
            missing_should.append(field)
            continue
        # otherwise skip
        continue

    # Validate email-like fields for basic format if present
    invalid_email_fields: List[str] = []
    for f in list(final.keys()):
        if _looks_like_email_field(f):
            norm = _normalize_email_list(final.get(f))
            if not norm and not _is_empty(final.get(f)):
                invalid_email_fields.append(f)
            else:
                final[f] = norm if norm else final.get(f)

    if invalid_email_fields:
        if len(invalid_email_fields) == 1:
            return False, f"Could you share a valid email address for {invalid_email_fields[0]}?", final
        else:
            return False, f"Could you share valid email addresses for {', '.join(invalid_email_fields)}?", final

    if missing_should:
        priority = ["attendees", "email", "emails", "invitees", "phone", "contact", "name"]
        chosen = None
        for p in priority:
            for m in missing_should:
                if p in m.lower():
                    chosen = m
                    break
            if chosen:
                break
        if not chosen:
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
    # final end_date_time fill
    if ok and "start_date_time" in final_payload and _is_empty(final_payload.get("end_date_time")) and "end_date_time" in template_obj:
        start_val = final_payload.get("start_date_time")
        if isinstance(start_val, str):
            try:
                dt = datetime.fromisoformat(start_val)
                final_payload["end_date_time"] = (dt + timedelta(minutes=30)).isoformat()
            except Exception:
                pass
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
            def make_api_call(state: Annotated[dict, InjectedState], **kwargs) -> str:
                event_id = str(uuid.uuid4())
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
                # Validate using the new resolver (passes conversation and agent prompt)
                ok, question, payload2 = _validate_payload_with_template_and_askmap(payload, _tpl_obj, _ask_map, conversation_for_context, agent_prompt_for_context)
                if not ok:
                    tool_result = {"ok": False, "status_code": None, "response": None, "event_id": event_id, "needs_input": True, "question": question}
                    logger.log("tool.validation", f"api_tool_{_tool_id} needs user input", tool_result)
                    return json.dumps(tool_result, ensure_ascii=False)
                if not _is_valid_api_url(_api_url):
                    error_data = {"ok": False, "status_code": None, "response": None, "event_id": event_id, "error": "Invalid api_url"}
                    logger.log("tool.error", f"api_tool_{_tool_id} invalid api_url", error_data)
                    return json.dumps(error_data, ensure_ascii=False)
                try:
                    if isinstance(payload2, dict):
                        context_vars: Dict[str, Any] = {}
                        existing_ctx = payload2.get("context_variables")
                        if isinstance(existing_ctx, dict):
                            context_vars.update(existing_ctx)
                        if isinstance(current_vars, dict):
                            context_vars.update(current_vars)
                        payload2["context_variables"] = context_vars
                    resp = requests.post(_api_url, json=payload2, timeout=20)
                    try:
                        response_data = resp.json()
                    except Exception:
                        response_data = resp.text
                    tool_result = {"ok": bool(resp.ok), "status_code": resp.status_code, "response": response_data, "event_id": event_id}
                    logger.log("tool.response", f"api_tool_{_tool_id} result", tool_result)
                    return json.dumps(tool_result, ensure_ascii=False)
                except Exception as e:
                    error_data = {"ok": False, "status_code": None, "response": None, "event_id": event_id, "error": str(e)}
                    logger.log("tool.error", f"api_tool_{_tool_id} failed", error_data)
                    return json.dumps(error_data, ensure_ascii=False)
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
def run_agent(agent_id: str, conversation_history: List[Dict[str, Any]], message: str, variables: Optional[Any] = None) -> Dict[str, Any]:
    logger.clear()
    logger.log("run.start", "Agent started", {"input_agent_id": agent_id})
    initial_vars = _variables_array_to_dict(variables)
    agent_resp = fetch_agent_details(agent_id)
    if not agent_resp:
        logger.log("run.error", "Agent details fetch returned None", {"agent_id": agent_id})
        return {"reply": "Error: Failed to fetch agent details.", "logs": logger.to_list(), "variables": _variables_dict_to_array(initial_vars)}
    api_key_to_use = None
    for key_name in ("api_key", "openai_api_key", "openai_key", "key"):
        if agent_resp.get(key_name):
            api_key_to_use = agent_resp.get(key_name)
            break
    if not api_key_to_use:
        api_key_to_use = os.getenv("OPENAI_API_KEY")
    if not api_key_to_use:
        logger.log("run.error", "OpenAI API key missing", {})
        return {"reply": "Error: Missing OpenAI API key.", "logs": logger.to_list(), "variables": _variables_dict_to_array(initial_vars)}
    agent_prompt = str(agent_resp.get("prompt", "You are a helpful assistant.") or "").strip()
    system_prompt = (f"{agent_prompt}\nTool rules:\nIf a tool returns JSON with needs_input=true and a question field, ask that single question to the user and stop.\nDo not claim an external action succeeded unless a tool result clearly confirms it.\nDo not invent missing user details.\n"
        "Runtime rulebook for planning and tool calls.\n"
        "You are the runtime planner and reply generator. Before proposing any external action, read the tool's INSTRUCTIONS and AskGuidance tags. For every field marked AskGuidance=SHOULD_BE_ASKED you must either have that exact value provided by the user earlier in the conversation or ask the user a single clear question requesting it. Do not invent or guess values for SHOULD_BE_ASKED fields. If any SHOULD_BE_ASKED field is missing or unclear, output a single question asking for that field and stop. Your output must follow the planner JSON schema and must never call a tool until all SHOULD_BE_ASKED fields are satisfied.\n"
        "INSTRCUTION SEPERATES THE PROPS BASED ON NEED OF ASKING THE PROP'S VALUE OR REFFERENCE FROM THE USER: SHOULD_BE_ASKED CAN_BE_ASKED AND NOT_TO_BE_ASKED, Always ask the Should be Asked FIleds, Never generally Can be asked can be asked from the user, only ask if it is important in the usecase or to compelte the action, NEVER ask the not to be asked fields.\n"
        "When asking, ask exactly one question that requests only the missing information and include the exact field name the system expects.\n"
        "When running a tool, ensure the payload structure matches the tool payload template exactly. If a tool returns a 'needs_input' response, surface that question to the user immediately and stop.\n"
        "Do not claim success unless a tool response confirms success in JSON. Keep replies user-facing and concise only after tools confirm success.\n"
    )
    def _render_system_prompt(current_vars: Dict[str, Any]) -> str:
        try:
            vars_str = json.dumps(current_vars, ensure_ascii=False)
        except Exception:
            vars_str = str(current_vars)
        return f"{system_prompt}\nCURRENT AGENT VARIABLES:\n{vars_str}"

    def _state_modifier_fn(state: AgentState) -> Dict[str, Any]:
        current_vars: Dict[str, Any] = {}
        try:
            if isinstance(state, dict):
                current_vars = state.get("variables", {}) or {}
        except Exception:
            current_vars = {}
        system_msg = SystemMessage(content=_render_system_prompt(current_vars))
        messages = []
        base_state: Dict[str, Any] = {}
        if isinstance(state, dict):
            base_state = dict(state)
            messages = base_state.get("messages", []) or []
        # Avoid stacking multiple system messages across steps
        filtered = [m for m in messages if not (isinstance(m, SystemMessage) and "CURRENT AGENT VARIABLES:" in m.content)]
        base_state["messages"] = [system_msg] + filtered
        base_state["variables"] = current_vars
        if "is_last_step" not in base_state:
            base_state["is_last_step"] = False
        return base_state
    fetched_tools = fetch_agent_tools(str(agent_id))
    merged_config = dict(DYNAMIC_CONFIG)
    if isinstance(fetched_tools, dict):
        for tid, tcfg in fetched_tools.items():
            merged_config[str(tid)] = {"api_url": tcfg.get("api_url", ""), "api_payload_json": tcfg.get("api_payload_json", ""), "instructions": tcfg.get("instructions", ""), "when_run": tcfg.get("when_run", "")}
    logger.log("run.config", "Merged dynamic config", {"tool_count": len(merged_config)})
    tools = [MANAGE_VARIABLES_TOOL] + create_universal_tools(merged_config)
    llm = ChatOpenAI(api_key=api_key_to_use, model="gpt-4o", temperature=0)
    # Build agent
    agent = create_react_agent(llm, tools, state_modifier=_state_modifier_fn, state_schema=AgentState)
    msgs = _to_messages(conversation_history, message)

    # Inject runtime context globals so tool calls can access conversation and agent prompt deterministically.
    # The resolver prefers explicit _conversation and _agent_prompt kwargs, otherwise falls back to these globals.
    globals()["_CURRENT_RUNTIME_CONVERSATION"] = (conversation_history or []) + [{"role": "user", "content": message}]
    globals()["_CURRENT_AGENT_PROMPT"] = agent_prompt

    try:
        # Log before model call
        logger.log("agent.invoke.start", "Invoking agent", {"message_count": len(msgs)})
        state = agent.invoke({"messages": msgs, "variables": initial_vars, "is_last_step": False}, config={"recursion_limit": 50})
        logger.log("agent.invoke.end", "Agent finished invoke")
    except GraphRecursionError as ge:
        # Root-level safeguard: if the graph keeps looping, fall back to a single-shot LLM response
        logger.log("run.error", "Graph recursion limit hit, falling back to direct reply", {"error": str(ge)})
        fallback_prompt = _render_system_prompt(initial_vars)
        try:
            fallback_msgs = [SystemMessage(content=fallback_prompt)] + msgs
            fallback_resp = llm.invoke(fallback_msgs)
            reply_text = fallback_resp.content if hasattr(fallback_resp, "content") else str(fallback_resp)
        except Exception as le:
            logger.log("run.error", "Fallback LLM failed", {"error": str(le)})
            reply_text = f"Error: {str(ge)}"
        return {"reply": reply_text, "logs": logger.to_list(), "variables": _variables_dict_to_array(initial_vars)}
    except Exception as e:
        logger.log("run.error", "Agent execution exception", {"error": str(e), "traceback": traceback.format_exc()})
        return {"reply": f"Error: {str(e)}", "logs": logger.to_list(), "variables": _variables_dict_to_array(initial_vars)}
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
            reply_text = "Done."
    except Exception:
        reply_text = "Done."
    final_variables_dict = initial_vars
    try:
        if isinstance(state, dict) and isinstance(state.get("variables"), dict):
            final_variables_dict = state.get("variables", initial_vars) or initial_vars
    except Exception:
        final_variables_dict = initial_vars
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
                        return {"reply": str(q), "logs": logger.to_list(), "variables": _variables_dict_to_array(final_variables_dict)}
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
                compact_trace.append({"role": "tool", "content": (m.content[:700] + "…") if isinstance(m.content, str) and len(m.content) > 700 else m.content})
            elif isinstance(m, AIMessage):
                tc = getattr(m, "tool_calls", None)
                compact_trace.append({"role": "assistant", "content": (m.content[:500] + "…") if isinstance(m.content, str) and len(m.content) > 500 else m.content, "tool_calls": tc})
        logger.log("run.trace", "Final message trace (compact)", {"messages": compact_trace})
    except Exception:
        pass
    return {"reply": reply_text, "logs": logger.to_list(), "variables": _variables_dict_to_array(final_variables_dict)}

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI()
# In dev allow all origins. In production set explicit origins list.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.post("/run-agent")
async def run_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"reply": "Error: Invalid JSON request body", "logs": logger.to_list()})
    agent_id = body.get("agent_id") or body.get("agentId") or body.get("id")
    if not agent_id:
        return JSONResponse({"reply": "Error: Missing agent_id in request", "logs": logger.to_list()})
    message = body.get("message", "")
    conversation = body.get("conversation", [])
    variables_payload = body.get("variables", [])
    res = run_agent(str(agent_id), conversation, str(message), variables_payload)
    return JSONResponse(res)

# Support inline execution when running inside CI or sandbox that injects 'inputs'
if "inputs" in globals():
    data = globals().get("inputs", {})
    agent_id = data.get("agent_id") or data.get("agentId") or data.get("id")
    _out = run_agent(str(agent_id), data.get("conversation", []), data.get("message", ""), data.get("variables", []))
    globals()["result"] = _out
