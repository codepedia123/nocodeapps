# main.py - LangGraph create_react_agent runtime with dynamic tools (Redis fetched), reply + logs
import os
import json
import uuid
import traceback
import urllib.parse
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
import requests

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from upstash_redis import Redis as UpstashRedis
from langgraph.prebuilt import create_react_agent

from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from pydantic import create_model, Field

# Minimal dynamic config
DYNAMIC_CONFIG: Dict[str, Any] = {}

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

def _parse_ask_guidance(instructions_text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not instructions_text:
        return out
    for m in re.finditer(r"'([^']+)'.*?AskGuidance=([A-Z_]+)", instructions_text):
        field_path = m.group(1).strip()
        guidance = m.group(2).strip()
        if field_path:
            out[field_path] = guidance
    return out

def _validate_payload_with_template_and_askmap(payload: Dict[str, Any], template_obj: Any, ask_map: Dict[str, str]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if not isinstance(payload, dict):
        return False, "Payload must be a JSON object.", {}
    if not isinstance(template_obj, dict):
        return True, None, payload
    for k in template_obj.keys():
        if k not in payload:
            return False, f"Could you share the value for '{k}' to proceed?", payload
    if "start_date_time" in template_obj and "end_date_time" in template_obj and _is_empty(payload.get("end_date_time")):
        start_val = payload.get("start_date_time")
        if isinstance(start_val, str):
            try:
                dt = datetime.fromisoformat(start_val)
                payload["end_date_time"] = (dt + timedelta(minutes=30)).isoformat()
            except Exception:
                pass
    missing_required: List[str] = []
    invalid_email_fields: List[str] = []
    for field_path, guidance in (ask_map or {}).items():
        if guidance != "SHOULD_BE_ASKED":
            continue
        parts = field_path.split(".")
        cur: Any = payload
        for p in parts:
            if not isinstance(cur, dict) or p not in cur:
                cur = None
                break
            cur = cur[p]
        if _is_empty(cur):
            missing_required.append(field_path)
            continue
        if _looks_like_email_field(parts[-1]):
            emails = _normalize_email_list(cur)
            if not emails:
                invalid_email_fields.append(field_path)
    if invalid_email_fields:
        return False, "Please share a valid email address to proceed.", payload
    if missing_required:
        if len(missing_required) == 1:
            return False, f"Could you share {missing_required[0]} to proceed?", payload
        return False, f"Could you share {', '.join(missing_required[:-1])} and {missing_required[-1]} to proceed?", payload
    return True, None, payload

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
            def make_api_call(**kwargs) -> str:
                event_id = str(uuid.uuid4())
                payload = dict(kwargs or {})
                logger.log("tool.call", f"api_tool_{_tool_id} triggered", {"event_id": event_id, "api_url": _api_url, "payload": payload})
                # Validate
                ok, question, payload2 = _validate_payload_with_template_and_askmap(payload, _tpl_obj, _ask_map)
                if not ok:
                    tool_result = {"ok": False, "status_code": None, "response": None, "event_id": event_id, "needs_input": True, "question": question}
                    logger.log("tool.validation", f"api_tool_{_tool_id} needs user input", tool_result)
                    return json.dumps(tool_result, ensure_ascii=False)
                if not _is_valid_api_url(_api_url):
                    error_data = {"ok": False, "status_code": None, "response": None, "event_id": event_id, "error": "Invalid api_url"}
                    logger.log("tool.error", f"api_tool_{_tool_id} invalid api_url", error_data)
                    return json.dumps(error_data, ensure_ascii=False)
                try:
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
def run_agent(agent_id: str, conversation_history: List[Dict[str, Any]], message: str) -> Dict[str, Any]:
    logger.clear()
    logger.log("run.start", "Agent started", {"input_agent_id": agent_id})
    agent_resp = fetch_agent_details(agent_id)
    if not agent_resp:
        logger.log("run.error", "Agent details fetch returned None", {"agent_id": agent_id})
        return {"reply": "Error: Failed to fetch agent details.", "logs": logger.to_list()}
    api_key_to_use = None
    for key_name in ("api_key", "openai_api_key", "openai_key", "key"):
        if agent_resp.get(key_name):
            api_key_to_use = agent_resp.get(key_name)
            break
    if not api_key_to_use:
        api_key_to_use = os.getenv("OPENAI_API_KEY")
    if not api_key_to_use:
        logger.log("run.error", "OpenAI API key missing", {})
        return {"reply": "Error: Missing OpenAI API key.", "logs": logger.to_list()}
    agent_prompt = str(agent_resp.get("prompt", "You are a helpful assistant.") or "").strip()
    system_prompt = (f"{agent_prompt}\nTool rules:\nIf a tool returns JSON with needs_input=true and a question field, ask that single question to the user and stop.\nDo not claim an external action succeeded unless a tool result clearly confirms it.\nDo not invent missing user details.\n"
        "Runtime rulebook for planning and tool calls.\n"
        "You are the runtime planner and reply generator. Before proposing any external action, read the tool's INSTRUCTIONS and AskGuidance tags. For every field marked AskGuidance=SHOULD_BE_ASKED you must either have that exact value provided by the user earlier in the conversation or ask the user a single clear question requesting it. Do not invent or guess values for SHOULD_BE_ASKED fields. If any SHOULD_BE_ASKED field is missing or unclear, output a single question asking for that field and stop. Your output must follow the planner JSON schema and must never call a tool until all SHOULD_BE_ASKED fields are satisfied.\n"
        "If a field is AskGuidance=CAN_BE_ASKED, you may ask it only if it meaningfully changes the action. If a field is optional, you have to think yourslef, because in instructions, they mark only which are schema based requried or not, but have to think if the prop is case based required. FOR EXAMPLE, IN THE CALANDER EVEN CREATION ACTION IF WE TAKE: Title is makred as required, but you would never ask the Title of an even from the user, whereas email is makred as can be asked, whereas in out case, it needs to be asked. Think accordingaly. Which props to not ask from the user, which to not ask and let the system geenrate it's value, and which to totally omit. \n"
        "When asking, ask exactly one question that requests only the missing information and include the exact field name the system expects.\n"
        "When running a tool, ensure the payload structure matches the tool payload template exactly. If a tool returns a 'needs_input' response, surface that question to the user immediately and stop.\n"
        "Do not claim success unless a tool response confirms success in JSON. Keep replies user-facing and concise only after tools confirm success.\n"
    )
    fetched_tools = fetch_agent_tools(str(agent_id))
    merged_config = dict(DYNAMIC_CONFIG)
    if isinstance(fetched_tools, dict):
        for tid, tcfg in fetched_tools.items():
            merged_config[str(tid)] = {"api_url": tcfg.get("api_url", ""), "api_payload_json": tcfg.get("api_payload_json", ""), "instructions": tcfg.get("instructions", ""), "when_run": tcfg.get("when_run", "")}
    logger.log("run.config", "Merged dynamic config", {"tool_count": len(merged_config)})
    tools = create_universal_tools(merged_config)
    llm = ChatOpenAI(api_key=api_key_to_use, model="gpt-4o-mini", temperature=0)
    # Build agent
    agent = create_react_agent(llm, tools, state_modifier=system_prompt)
    msgs = _to_messages(conversation_history, message)
    try:
        # Log before model call
        logger.log("agent.invoke.start", "Invoking agent", {"message_count": len(msgs)})
        state = agent.invoke({"messages": msgs})
        logger.log("agent.invoke.end", "Agent finished invoke")
    except Exception as e:
        logger.log("run.error", "Agent execution exception", {"error": str(e), "traceback": traceback.format_exc()})
        return {"reply": f"Error: {str(e)}", "logs": logger.to_list()}
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
                        return {"reply": str(q), "logs": logger.to_list()}
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
    return {"reply": reply_text, "logs": logger.to_list()}

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
    res = run_agent(str(agent_id), conversation, str(message))
    return JSONResponse(res)

# Support inline execution when running inside CI or sandbox that injects 'inputs'
if "inputs" in globals():
    data = globals().get("inputs", {})
    agent_id = data.get("agent_id") or data.get("agentId") or data.get("id")
    _out = run_agent(str(agent_id), data.get("conversation", []), data.get("message", ""))
    globals()["result"] = _out
