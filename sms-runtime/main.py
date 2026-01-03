# main.py - LangChain v1 create_agent runtime with dynamic tools (Redis fetched), reply + logs
import os
import json
import uuid
import time
import traceback
from fastapi.middleware.cors import CORSMiddleware
import urllib.parse
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Upstash Redis (HTTP SDK)
from upstash_redis import Redis as UpstashRedis
from langgraph.prebuilt import create_react_agent


# LangChain v1 agent loop


from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from langchain_openai import ChatOpenAI

# Pydantic dynamic schemas for tool args
from pydantic import create_model, Field


# ---------------------------
# Configuration JSON
# ---------------------------
# Minimal default config, merged with fetched tools at runtime.
DYNAMIC_CONFIG: Dict[str, Any] = {}


# ---------------------------
# Logger helper
# ---------------------------
class Logger:
    def __init__(self):
        self._events: List[Dict[str, Any]] = []

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def log(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        entry = {
            "ts": self._now(),
            "id": str(uuid.uuid4()),
            "type": event_type,
            "message": message,
            "data": data or {},
        }
        self._events.append(entry)

    def to_list(self) -> List[Dict[str, Any]]:
        return self._events.copy()

    def clear(self):
        self._events = []


logger = Logger()


# ---------------------------
# Redis init (Upstash)
# ---------------------------
_redis_client: Optional[UpstashRedis] = None

try:
    redis_url = os.getenv("UPSTASH_REDIS_REST_URL", "https://climbing-hyena-56303.upstash.io")
    redis_token = os.getenv(
        "UPSTASH_REDIS_REST_TOKEN",
        "AdvvAAIncDExZmMzYTBiNTJhZWU0MzA1YjA1M2IwYWU4NThlZjcyM3AxNTYzMDM",
    )

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
# Redis fetch helpers (unchanged behavior)
# ---------------------------
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
        candidates: List[str] = []
        if isinstance(agent_id, str):
            candidates.append(agent_id)
            if agent_id.startswith("agent"):
                try:
                    suffix = int(agent_id[len("agent") :])
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

            key = cand if cand.startswith("agent") else f"agent{cand}" if cand.isdigit() else cand

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
                logger.log(
                    "fetch.agent.redis.error",
                    "Error checking agent key in Redis",
                    {"candidate": key, "error": str(e)},
                )
                continue

        logger.log(
            "fetch.agent.redis.notfound",
            "Agent not found in Redis",
            {"agent_id": agent_id, "candidates_tried": list(seen)},
        )
        return None

    except Exception as e:
        logger.log(
            "fetch.agent.redis.exception",
            "Unexpected error fetching agent",
            {"agent_id": agent_id, "error": str(e)},
        )
        return None


def fetch_agent_tools(agent_user_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch all rows from dynamic table 'all-agents-tools' directly from Redis only.
    Filter rows where row['agent_id'] matches agent_user_id (accepts '3' vs 'agent3').
    Return a dict of tool_id -> tool_definition or None if Redis unavailable.
    """
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
# Dynamic Tool Factory (with optional validation hints)
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
    return (
        val is None
        or (isinstance(val, str) and val.strip() == "")
        or (isinstance(val, (list, dict)) and len(val) == 0)
    )


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


def _validate_payload_with_template_and_askmap(
    payload: Dict[str, Any],
    template_obj: Any,
    ask_map: Dict[str, str],
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Generic validation:
    - Ensure top level structure keys exist if template is dict.
    - Enforce non empty values for AskGuidance=SHOULD_BE_ASKED.
    - Email like fields: require at least one valid email when SHOULD_BE_ASKED.
    - If template has start_date_time and end_date_time: auto fill end if missing (start + 30m) when parseable.
    Returns (ok, question, payload).
    """
    if not isinstance(payload, dict):
        return False, "Payload must be a JSON object.", {}

    if not isinstance(template_obj, dict):
        return True, None, payload

    # Ensure structure keys exist
    for k in template_obj.keys():
        if k not in payload:
            return False, f"Could you share the value for '{k}' to proceed?", payload

    # Auto fill end_date_time if useful
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

        # Email validation
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


def create_universal_tools(config: Dict[str, Any]) -> List[StructuredTool]:
    """
    Build StructuredTool objects dynamically from config.
    Each tool calls POST api_url with a JSON body produced by the model.
    """
    generated_tools: List[StructuredTool] = []

    for tool_id, cfg in (config or {}).items():
        api_url = str(cfg.get("api_url", "") or "")
        instructions = str(cfg.get("instructions", "") or "")
        when_run = str(cfg.get("when_run", "") or "")
        raw_payload_template = urllib.parse.unquote(str(cfg.get("api_payload_json", "") or ""))

        tpl_obj = _safe_json_loads(raw_payload_template)
        ask_map = _parse_ask_guidance(instructions)

        # Schema: keys from template, required so model "sees" exact fields.
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
                logger.log(
                    "tool.call",
                    f"api_tool_{_tool_id} triggered",
                    {"event_id": event_id, "api_url": _api_url, "payload": payload},
                )

                # Optional validation pass before network call
                ok, question, payload2 = _validate_payload_with_template_and_askmap(payload, _tpl_obj, _ask_map)
                if not ok:
                    tool_result = {
                        "ok": False,
                        "status_code": None,
                        "response": None,
                        "event_id": event_id,
                        "needs_input": True,
                        "question": question or "Could you share the missing details needed to proceed?",
                    }
                    logger.log("tool.validation", f"api_tool_{_tool_id} needs user input", tool_result)
                    return json.dumps(tool_result, ensure_ascii=False)

                # Network call
                try:
                    resp = requests.post(_api_url, json=payload2, timeout=20)
                    try:
                        response_data = resp.json()
                    except Exception:
                        response_data = resp.text

                    tool_result = {
                        "ok": bool(resp.ok),
                        "status_code": resp.status_code,
                        "response": response_data,
                        "event_id": event_id,
                    }
                    logger.log("tool.response", f"api_tool_{_tool_id} result", tool_result)
                    return json.dumps(tool_result, ensure_ascii=False)

                except Exception as e:
                    error_data = {
                        "ok": False,
                        "status_code": None,
                        "response": None,
                        "event_id": event_id,
                        "error": str(e),
                    }
                    logger.log("tool.error", f"api_tool_{_tool_id} failed", error_data)
                    return json.dumps(error_data, ensure_ascii=False)

            return make_api_call

        make_api_call = _make_api_call_factory(str(tool_id), api_url, tpl_obj, ask_map)

        description = (
            f"WHEN_RUN: {when_run}\n"
            f"INSTRUCTIONS: {instructions}\n"
            f"PAYLOAD_TEMPLATE (must match structure exactly): {raw_payload_template}\n"
            "IMPORTANT: Do not invent missing user details. If unknown, ask the user a single clear question."
        )

        new_tool = StructuredTool.from_function(
            func=make_api_call,
            name=f"sync_data_tool_{tool_id}",
            description=description,
            args_schema=DynamicArgsModel,
        )
        generated_tools.append(new_tool)

    return generated_tools


# ---------------------------
# Middleware for logging and tool errors
# ---------------------------
@before_agent
def mw_before_agent(state, runtime):
    logger.log("agent.before", "Agent invocation started", {"message_count": len(state.get("messages", []))})
    return None


@before_model
def mw_before_model(state, runtime):
    logger.log("model.before", "About to call model", {"message_count": len(state.get("messages", []))})
    return None


@after_model
def mw_after_model(state, runtime):
    msgs = state.get("messages", [])
    last = msgs[-1] if msgs else None
    data: Dict[str, Any] = {}
    if isinstance(last, AIMessage):
        data["has_tool_calls"] = bool(getattr(last, "tool_calls", None))
        data["content_preview"] = (last.content[:400] + "…") if isinstance(last.content, str) and len(last.content) > 400 else last.content
    logger.log("model.after", "Model returned", data)
    return None


@after_agent
def mw_after_agent(state, runtime):
    logger.log("agent.after", "Agent invocation finished", {"message_count": len(state.get("messages", []))})
    return None


@wrap_tool_call
def mw_tool_errors_and_logging(request, handler):
    """
    Official hook to handle tool failures and also log tool calls.
    If a tool raises, return a ToolMessage so the model can recover and ask the user.
    """
    try:
        tool_name = getattr(request, "tool", None).name if getattr(request, "tool", None) else None
        call = getattr(request, "tool_call", None) or {}
        logger.log(
            "tool.middleware.start",
            "Tool execution started",
            {"tool_name": tool_name, "tool_call_id": call.get("id"), "args": call.get("args")},
        )
        out = handler(request)
        logger.log(
            "tool.middleware.end",
            "Tool execution finished",
            {"tool_name": tool_name, "tool_call_id": call.get("id")},
        )
        return out
    except Exception as e:
        call = getattr(request, "tool_call", None) or {}
        logger.log(
            "tool.middleware.error",
            "Tool execution exception",
            {"tool_call_id": call.get("id"), "error": str(e), "traceback": traceback.format_exc()},
        )
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=call.get("id"),
        )


# ---------------------------
# Core Execution logic (create_agent loop)
# ---------------------------
def _to_messages(conversation_history: List[Dict[str, Any]], user_message: str) -> List[Any]:
    msgs: List[Any] = []

    # Convert provided history
    for turn in (conversation_history or []):
        role = str(turn.get("role", "") or "").lower().strip()
        content = turn.get("content")
        if content is None:
            content = turn.get("message")
        if content is None:
            content = ""

        if role == "user":
            msgs.append(HumanMessage(content=str(content)))
        elif role == "assistant":
            msgs.append(AIMessage(content=str(content)))
        else:
            # Unknown roles treated as user to avoid losing info
            msgs.append(HumanMessage(content=str(content)))

    # Append current user message
    if user_message is not None:
        msgs.append(HumanMessage(content=str(user_message)))

    return msgs


def run_agent(agent_id: str, conversation_history: List[Dict[str, Any]], message: str) -> Dict[str, Any]:
    """
    New runtime:
    - Fetch agent + tools from Redis
    - Create tools dynamically
    - Run LangChain v1 create_agent loop (single model does planning + replies + tool usage)
    - Return reply + logs
    """
    logger.clear()
    logger.log("run.start", "Agent started", {"input_agent_id": agent_id})

    # 1. Fetch agent details
    agent_resp = fetch_agent_details(agent_id)
    if not agent_resp:
        logger.log("run.error", "Agent details fetch returned None", {"agent_id": agent_id})
        return {"reply": "Error: Failed to fetch agent details.", "logs": logger.to_list()}

    # 2. API key selection
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

    # 3. Prompt
    agent_prompt = str(agent_resp.get("prompt", "You are a helpful assistant.") or "").strip()

    # Extra rule to make tool behavior predictable
    system_prompt = (
        f"{agent_prompt}\n\n"
        "Tool rules:\n"
        "If a tool returns JSON with needs_input=true and a question field, ask that single question to the user and stop.\n"
        "Do not claim an external action succeeded unless a tool result clearly confirms it.\n"
        "Do not invent missing user details.\n"
    )

    # 4. Fetch tools and merge config
    fetched_tools = fetch_agent_tools(str(agent_id))
    merged_config = dict(DYNAMIC_CONFIG)

    if isinstance(fetched_tools, dict):
        for tid, tcfg in fetched_tools.items():
            merged_config[str(tid)] = {
                "api_url": tcfg.get("api_url", ""),
                "api_payload_json": tcfg.get("api_payload_json", ""),
                "instructions": tcfg.get("instructions", ""),
                "when_run": tcfg.get("when_run", ""),
            }
    else:
        logger.log("run.tools", "No tools fetched from Redis, using default config", {})

    logger.log("run.config", "Merged dynamic config", {"tool_count": len(merged_config)})

    # 5. Create tools
    tools = create_universal_tools(merged_config)

    # 6. Build model
    llm = ChatOpenAI(api_key=api_key_to_use, model="gpt-4o-mini", temperature=0)

    # 7. Create official agent graph with middleware for limits + logging + tool errors
    agent = create_react_agent(
    llm,
    tools,
    state_modifier=system_prompt
    )


    # 8. Run agent
    msgs = _to_messages(conversation_history, message)

    try:
        state = agent_graph.invoke({"messages": msgs})
    except Exception as e:
        logger.log("run.error", "Agent execution exception", {"error": str(e), "traceback": traceback.format_exc()})
        return {"reply": f"Error: {str(e)}", "logs": logger.to_list()}

    # 9. Extract final reply (last AIMessage content)
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

    # 10. Also log message trace for debugging (compact)
    try:
        out_msgs = state.get("messages", []) if isinstance(state, dict) else []
        compact_trace: List[Dict[str, Any]] = []
        for m in out_msgs[-30:]:
            if isinstance(m, HumanMessage):
                compact_trace.append({"role": "user", "content": (m.content[:500] + "…") if len(m.content) > 500 else m.content})
            elif isinstance(m, ToolMessage):
                compact_trace.append({"role": "tool", "content": (m.content[:700] + "…") if len(m.content) > 700 else m.content})
            elif isinstance(m, AIMessage):
                tc = getattr(m, "tool_calls", None)
                compact_trace.append(
                    {
                        "role": "assistant",
                        "content": (m.content[:500] + "…") if isinstance(m.content, str) and len(m.content) > 500 else m.content,
                        "tool_calls": tc,
                    }
                )
        logger.log("run.trace", "Final message trace (compact)", {"messages": compact_trace})
    except Exception:
        pass

    return {"reply": reply_text, "logs": logger.to_list()}


# ---------------------------
# API Wrapper
# ---------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/run-agent")
async def run_endpoint(request: Request):
    """
    Expected request JSON:
    {
        "agent_id": "agent1",
        "message": "Hello",
        "conversation": [ {"role":"user","content":"..."} ]
    }
    Returns:
    {
        "reply": "...",
        "logs": [...]
    }
    """
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


# Support for interactive runtime evaluation similar to original script
if "inputs" in globals():
    data = globals().get("inputs", {})
    agent_id = data.get("agent_id") or data.get("agentId") or data.get("id")
    _out = run_agent(str(agent_id), data.get("conversation", []), data.get("message", ""))
    globals()["result"] = _out
