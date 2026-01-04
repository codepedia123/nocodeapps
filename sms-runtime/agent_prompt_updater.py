import os
import json
import traceback
import re
from typing import List, Dict, Any, Optional, Annotated, TypedDict
from urllib import request as urllib_request
from urllib.error import URLError, HTTPError

try:
    import requests  # type: ignore
except Exception:
    requests = None  # Fallback to urllib when requests is unavailable
from pydantic import BaseModel, Field

# LangChain & LangGraph Imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

# ------------------------------------------------------------------
# 1. SETUP INPUTS (Unified Logic)
# ------------------------------------------------------------------
_raw_globals = globals()
data: Dict[str, Any] = {}

for key in ("inputs", "input", "payload"):
    candidate = _raw_globals.get(key)
    if isinstance(candidate, dict) and candidate:
        data.update(candidate)

for k in ("long_text", "message", "conversation", "api_key", "apiKey", "openai_api_key", "key", "agent_id", "agentId", "agent"):
    if k in _raw_globals and _raw_globals.get(k) not in (None, "", {}):
        if k not in data:
            data[k] = _raw_globals.get(k)

# Env Var Fallback for JSON
if not data:
    for env_key in ("REQUEST_BODY", "RAW_BODY", "INPUT_JSON"):
        raw = os.getenv(env_key)
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    data.update(parsed)
                    break
            except: pass
else:
    # Secondary env fallback for agent id if provided separately
    if "agent_id" not in data:
        env_agent = os.getenv("AGENT_ID")
        if env_agent:
            data["agent_id"] = env_agent

long_text = data.get("long_text", "")
user_message = data.get("message", "") or data.get("input_message", "") or data.get("msg", "")
conversation = data.get("conversation", []) or data.get("chat_history", [])
api_key_to_use = (data.get("api_key") or data.get("openai_api_key") or data.get("apiKey") or os.getenv("OPENAI_API_KEY"))

# Helper utilities for tool management
def _extract_agent_id_from_context() -> Optional[str]:
    """
    Attempts to locate an agent id in the form 'agent7' or 'agent 7' from
    known inputs. Returns the numeric portion as a string, or None if missing.
    """
    candidates: List[str] = []
    for key in ("agent_id", "agentId", "agent"):
        val = data.get(key)
        if isinstance(val, str):
            candidates.append(val)
        elif isinstance(val, (int, float)):
            candidates.append(str(val))

    # Search the user message and long_text for patterns like 'agent7'
    candidates.extend([user_message, long_text])
    for turn in conversation:
        content = turn.get("content") or turn.get("message") or ""
        if isinstance(content, str):
            candidates.append(content)

    pattern = re.compile(r"agent\s*-?\s*(\d+)", flags=re.IGNORECASE)
    for cand in candidates:
        if not cand:
            continue
        m = pattern.search(str(cand))
        if m:
            return m.group(1)
    return None


def _split_when_run(raw: str) -> List[str]:
    """
    Splits a when_run string into a list. Primary delimiter is '|'.
    If no delimiter is found, returns a single-item list when text exists.
    """
    if not raw:
        return []
    if "|" in raw:
        parts = [p.strip() for p in raw.split("|") if p.strip()]
    else:
        parts = [raw.strip()] if raw.strip() else []
    return parts


def _http_get_json(url: str) -> Optional[Dict[str, Any]]:
    try:
        if requests:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        with urllib_request.urlopen(url, timeout=10) as resp:
            if resp.status >= 400:
                return None
            data_bytes = resp.read()
            return json.loads(data_bytes.decode("utf-8"))
    except (URLError, HTTPError, json.JSONDecodeError, Exception):
        return None


def _http_post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if requests:
            resp = requests.post(url, json=payload, timeout=10)
            return {"status": resp.status_code, "body": resp.text}
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"})
        with urllib_request.urlopen(req, timeout=10) as resp:
            return {"status": resp.status, "body": resp.read().decode("utf-8")}
    except Exception as e:
        return {"status": "error", "body": str(e)}


def _fetch_tools_for_agent(agent_id_number: Optional[str]) -> Dict[str, Any]:
    """
    Fetches all tools and returns:
    {
        "tools": [ { "tool_id": str, "piece_id": str, "when_run": [..], "raw_when_run": str } ],
        "error": Optional[str]
    }
    """
    if not agent_id_number:
        return {"tools": [], "error": "No agent id found in context."}

    url = "https://adequate-compassion-production.up.railway.app/fetch?table=all-agents-tools"
    payload = _http_get_json(url)
    if not isinstance(payload, dict):
        return {"tools": [], "error": "Failed to fetch tools for agent."}

    tools: List[Dict[str, Any]] = []
    for db_id, tool in payload.items():
        try:
            tool_agent_id = str(tool.get("agent_id", "")).strip()
        except Exception:
            tool_agent_id = ""
        if not tool_agent_id or tool_agent_id != str(agent_id_number):
            continue

        when_raw = tool.get("when_run") or ""
        tools.append({
            "tool_id": str(db_id),
            "piece_id": tool.get("piece-id") or tool.get("piece_id"),
            "when_run": _split_when_run(when_raw),
            "raw_when_run": when_raw
        })

    return {"tools": tools, "error": None}

# ------------------------------------------------------------------
# 2. DEFINE THE STATE & TOOLS
# ------------------------------------------------------------------

class AgentState(TypedDict):
    """The official way to track state across the agent's 'thought' process."""
    messages: Annotated[List[BaseMessage], "The conversation messages"]
    document: str
    change_summary: Optional[List[str]]
    is_updated: bool
    error_log: Optional[str]  # Added to track why an update might fail
    tools_catalog: Optional[List[Dict[str, Any]]]
    agent_id_number: Optional[str]

@tool
def patch_document_tool(original_snippet: str, replacement_text: str, explanation: str, anchor_for_insertion: Optional[str] = None):
    """
    Updates the document by replacing a specific snippet with new text.
    'explanation' should be a professional, descriptive summary of what you are changing.
    'anchor_for_insertion' (optional) can be provided when original_snippet is not present:
      - If provided and found, the replacement_text will be inserted after that anchor.
      - This tool itself does not mutate the document (the executor applies changes).
    Returns a stable, human-readable summary the agent can use.
    """
    return {
        "status": "planned",
        "action": "patch",
        "original_snippet": original_snippet,
        "replacement_text": replacement_text,
        "explanation": explanation,
        "anchor_for_insertion": anchor_for_insertion
    }

@tool
def insert_after_anchor_tool(anchor_snippet: str, insertion_text: str, explanation: str):
    """
    Requests insertion of insertion_text immediately after the first occurrence of anchor_snippet.
    The executor (tool_executor) will perform the insertion on the document.
    """
    return {
        "status": "planned",
        "action": "insert_after_anchor",
        "anchor_snippet": anchor_snippet,
        "insertion_text": insertion_text,
        "explanation": explanation
    }

@tool
def update_when_run_tool(tool_id: str, scenario_updates: List[Dict[str, Any]], explanation: str):
    """
    Requests updates to a tool's when_run scenarios for the current agent.
    Provide 'scenario_updates' as a list of {'index': 1-based index to replace or append, 'text': full replacement scenario}.
    Only request this when changing when_run is essential to satisfy the user's update.
    """
    return {
        "status": "planned",
        "action": "update_when_run",
        "tool_id": tool_id,
        "scenario_updates": scenario_updates,
        "explanation": explanation
    }

# ------------------------------------------------------------------
# 3. AGENT LOGIC UNIT
# ------------------------------------------------------------------

def run_updater_agent():
    if not api_key_to_use:
        return {"error": "Missing 'api_key' in request payload."}

    try:
        agent_id_number = _extract_agent_id_from_context()
        tool_fetch_result = _fetch_tools_for_agent(agent_id_number)
        tools_catalog = tool_fetch_result.get("tools") or []

        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key_to_use)
        tools = [patch_document_tool, insert_after_anchor_tool, update_when_run_tool]
        llm_with_tools = llm.bind_tools(tools)

        # 1. THE AGENT NODE
        def call_model(state: AgentState):
            tools_json = json.dumps(state.get("tools_catalog") or [], ensure_ascii=True)
            # Add explicit guidance to be conservative and prefer precise updates
            sys_msg = SystemMessage(content=(
                "You are a professional Editor.\n"
                "IMPORTANT EDITING RULES (READ CAREFULLY):\n"
                "1. Default to 'patch_document_tool' to update existing text precisely and keep overall length conservative.\n"
                "2. Use 'insert_after_anchor_tool' ONLY for entirely new insertions that are not already present in the document.\n"
                "3. If the exact snippet is not present, do NOT attempt to patch a demo chat transcript directly.\n"
                "   Instead, use 'patch_document_tool' with 'anchor_for_insertion' to place the new text after a stable anchor.\n"
                "4. Provide the EXACT original snippet when using 'patch_document_tool' whenever possible.\n"
                "5. The 'anchor_for_insertion' must be a short, guaranteed-to-exist heading or line such as '# Notes:' or '# Task:'.\n"
                "6. Your 'explanation' must be a clear, descriptive summary of the change.\n"
                "7. Keep changes minimal: avoid expanding the prompt unless the user explicitly asks for more detail.\n"
                "8. If content already exists in the document, update it instead of inserting a duplicate.\n\n"
                "TOOLS CATALOG (agent {agent}):\n{tools}\n\n"
                "WHEN_RUN UPDATE RULES:\n"
                "- Update a tool's when_run scenarios ONLY when it is necessary to fulfill the user's requested change.\n"
                "- Prefer editing existing scenarios; append a new one only when clearly required and with minimal wording.\n"
                "- Use the exact 'tool_id' from the catalog when calling 'update_when_run_tool'.\n"
                "- Call 'update_when_run_tool' with 1-based 'index' entries for every scenario you change; supply the full replacement text for each index.\n"
                "- Provide the full replacement scenario text for every index you modify (no partial fragments).\n"
                "- If you need to add a scenario, use index = current_length + 1 (no gaps). Do not invent indices.\n"
                "- Only change when_run entries that align the tool trigger with the requested behavior; avoid unrelated changes.\n"
                "- Always include an 'explanation' describing why the when_run change is essential.\n\n"
                f"CURRENT DOCUMENT CONTENT:\n--- START ---\n{state['document']}\n--- END ---\n\n"
                "Answer format: If you want to call a tool, call the appropriate tool with JSON args. Otherwise reply normally.\n"
            ).format(agent=state.get("agent_id_number") or "unknown", tools=tools_json))
            response = llm_with_tools.invoke([sys_msg] + state["messages"])
            return {"messages": [response]}

        # 2. THE TOOL EXECUTION NODE (The Surgical Update)
        def tool_executor(state: AgentState):
            def _find_anchor_end(doc: str, anchor: str) -> Optional[int]:
                """
                Returns the insertion index right after the first matched anchor.
                Tries exact match first, then a relaxed match that tolerates whitespace differences,
                common markdown heading formatting differences, and case differences.
                """
                if not anchor:
                    return None

                # 1) Exact match
                idx = doc.find(anchor)
                if idx != -1:
                    return idx + len(anchor)

                a = anchor.strip()
                if not a:
                    return None

                # 2) Relaxed match
                # Special handling for markdown heading-like anchors, eg "#Notes:" vs "# Notes:"
                if a.startswith("#"):
                    core = a.lstrip("#").strip()
                    if not core:
                        return None
                    core_escaped = re.escape(core).replace(r"\ ", r"\s+")
                    pattern = rf"#+\s*{core_escaped}"
                else:
                    pattern = re.escape(a).replace(r"\ ", r"\s+")

                # If anchor included a colon, allow optional whitespace before it
                pattern = pattern.replace(r"\:", r"\s*:")

                m = re.search(pattern, doc, flags=re.IGNORECASE)
                if m:
                    return m.end()

                return None

            last_msg = state["messages"][-1]
            new_doc = state["document"]
            summary = state.get("change_summary") or []
            updated = state.get("is_updated", False)
            err = state.get("error_log")
            tools_catalog = state.get("tools_catalog") or []

            # Normalize potential tool calls representation
            tool_calls = []
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                tool_calls = last_msg.tool_calls
            elif isinstance(last_msg, dict) and last_msg.get("tool_calls"):
                tool_calls = last_msg.get("tool_calls")

            for tool_call in tool_calls:
                # tool_call may be a dict with 'name' & 'args'
                name = tool_call.get("name") if isinstance(tool_call, dict) else None
                args = tool_call.get("args", {}) if isinstance(tool_call, dict) else {}

                # Some runtimes nest args differently, handle both
                if not isinstance(args, dict) and hasattr(args, "__dict__"):
                    args = vars(args)

                def _append_summary(text: str):
                    if text:
                        summary.append(text)

                # Patch action
                if name == "patch_document_tool" or (isinstance(tool_call, dict) and tool_call.get("tool") == "patch_document_tool"):
                    old = args.get("original_snippet", "") or ""
                    new = args.get("replacement_text", "") or ""
                    expl = args.get("explanation", "") or ""
                    anchor = args.get("anchor_for_insertion", None)

                    if old and old in new_doc:
                        new_doc = new_doc.replace(old, new, 1)
                        _append_summary(expl or "Replaced an exact matching snippet.")
                        updated = True
                        err = None
                    else:
                        # fallback 1: if anchor provided, try relaxed anchor match and insert after anchor
                        if anchor:
                            end_idx = _find_anchor_end(new_doc, anchor)
                            if end_idx is not None:
                                insertion = f"\n{new}\n"
                                new_doc = new_doc[:end_idx] + insertion + new_doc[end_idx:]
                                _append_summary(expl or "Inserted new text after provided anchor.")
                                updated = True
                                err = None
                            else:
                                # fallback 2: true secondary fallback, append at end
                                new_doc = (new_doc.rstrip() + "\n\n" + new.strip() + "\n")
                                _append_summary(expl or "Anchor not found, appended new section at end.")
                                updated = True
                                err = None
                        else:
                            # fallback 2: true secondary fallback, append at end
                            new_doc = (new_doc.rstrip() + "\n\n" + new.strip() + "\n")
                            _append_summary(expl or "No snippet match and no anchor provided, appended new section at end.")
                            updated = True
                            err = None

                # Insert-after-anchor action
                elif name == "insert_after_anchor_tool" or (isinstance(tool_call, dict) and tool_call.get("tool") == "insert_after_anchor_tool"):
                    anchor = args.get("anchor_snippet", "") or ""
                    insertion_text = args.get("insertion_text", "") or ""
                    expl = args.get("explanation", "") or ""

                    if insertion_text and insertion_text in new_doc:
                        _append_summary(expl or "Insertion skipped because the text already exists.")
                        updated = True
                        err = None
                        continue

                    end_idx = _find_anchor_end(new_doc, anchor)
                    insertion = f"\n{insertion_text}\n"

                    if end_idx is not None:
                        new_doc = new_doc[:end_idx] + insertion + new_doc[end_idx:]
                        _append_summary(expl or "Inserted text after anchor.")
                        updated = True
                        err = None
                    else:
                        # true secondary fallback: append at end
                        new_doc = (new_doc.rstrip() + "\n\n" + insertion_text.strip() + "\n")
                        _append_summary(expl or "Anchor not found, appended new section at end.")
                        updated = True
                        err = None

                # Update when_run action
                elif name == "update_when_run_tool" or (isinstance(tool_call, dict) and tool_call.get("tool") == "update_when_run_tool"):
                    tool_id = str(args.get("tool_id") or "").strip()
                    scenario_updates = args.get("scenario_updates") or []
                    expl = args.get("explanation", "") or ""

                    if not tool_id:
                        err = "Missing tool_id for when_run update."
                        _append_summary(expl or err)
                        continue

                    if not isinstance(scenario_updates, list) or not scenario_updates:
                        err = f"No scenario_updates provided for tool {tool_id}."
                        _append_summary(expl or err)
                        continue

                    target_tool = next((t for t in tools_catalog if str(t.get("tool_id")) == tool_id), None)
                    if not target_tool:
                        err = f"Tool {tool_id} not found for current agent."
                        _append_summary(expl or err)
                        continue

                    current_when = list(target_tool.get("when_run") or [])
                    if not current_when and target_tool.get("raw_when_run"):
                        current_when = _split_when_run(target_tool.get("raw_when_run") or "")

                    changed = False
                    out_of_range = False
                    out_of_range_note = None
                    for upd in scenario_updates:
                        try:
                            idx = int(upd.get("index"))
                        except Exception:
                            continue
                        text = (upd.get("text") or "").strip()
                        if idx <= 0 or not text:
                            continue

                        if idx <= len(current_when):
                            current_when[idx - 1] = text
                            changed = True
                        elif idx == len(current_when) + 1:
                            current_when.append(text)
                            changed = True
                        else:
                            out_of_range = True
                            err = f"Cannot update when_run index {idx} (out of range)."
                            out_of_range_note = err

                    if not changed:
                        _append_summary(expl or "No when_run changes applied.")
                        continue

                    new_when_raw = "|".join(current_when)
                    target_tool["when_run"] = current_when
                    target_tool["raw_when_run"] = new_when_raw

                    update_payload = {
                        "table": "all-agents-tools",
                        "id": tool_id,
                        "updates": {"when_run": new_when_raw}
                    }
                    update_resp = _http_post_json("https://adequate-compassion-production.up.railway.app/update", update_payload)

                    resp_status = update_resp.get("status")
                    status_line = f"when_run for tool {tool_id} updated (status={resp_status})."
                    is_error_status = (resp_status == "error") or (isinstance(resp_status, int) and resp_status >= 400)
                    if is_error_status:
                        err = f"Update call failed for tool {tool_id}: {update_resp.get('body')}"
                        _append_summary(err)
                    else:
                        if out_of_range_note:
                            _append_summary(out_of_range_note)
                        _append_summary(expl or status_line)
                        updated = True
                        if out_of_range:
                            err = err or "One or more indices were out of range."
                        else:
                            err = None

                else:
                    # Unknown tool name: capture for diagnostics
                    err = f"Tool call with unknown tool name: {name}"
                    summary = summary or None

            return {
                "document": new_doc,
                "change_summary": summary,
                "is_updated": updated,
                "error_log": err,
                "tools_catalog": tools_catalog
            }

        # 3. BUILD THE GRAPH
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_executor)

        workflow.set_entry_point("agent")

        def should_continue(state: AgentState):
            # safe guard: messages may be lists of Message objects or dicts
            try:
                last = state["messages"][-1]
                if hasattr(last, "tool_calls") and last.tool_calls:
                    return "tools"
                if isinstance(last, dict) and last.get("tool_calls"):
                    return "tools"
            except Exception:
                pass
            return END

        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", END)

        app = workflow.compile()

        # 4. EXECUTE
        history = []
        for turn in conversation:
            content = turn.get("content", turn.get("message", ""))
            role = (turn.get("role") or "").lower()
            if "user" in role:
                history.append(HumanMessage(content=content))
            else:
                history.append(AIMessage(content=content))

        inputs = {
            "messages": history + [HumanMessage(content=user_message)],
            "document": long_text,
            "change_summary": [],
            "is_updated": False,
            "error_log": None,
            "tools_catalog": tools_catalog,
            "agent_id_number": agent_id_number
        }

        final_state = app.invoke(inputs)

        # ------------------------------------------------------------------
        # 5. CONSTRUCT FINAL RESPONSE (Improved Logic)
        # ------------------------------------------------------------------
        last_msg = final_state["messages"][-1] if final_state.get("messages") else None

        if final_state.get("is_updated"):
            # Scenario A: Tool worked perfectly
            summary = "; ".join(final_state.get("change_summary") or [])
            final_reply = f"I have updated the text as requested. {summary}".strip()
        elif final_state.get("error_log"):
            # Scenario B: Tool was called but the snippet didn't match
            summary = "; ".join(final_state.get("change_summary") or [])
            final_reply = (
                f"I tried to make the following change: '{summary}', but hit an issue: {final_state.get('error_log')}. "
                "For document edits, ensure the snippet or anchor (for example '# Notes:' or '# Task:') is present. "
                "For tool updates, verify the tool_id and 1-based when_run indices match the catalog."
            )
        else:
            # Scenario C: Chatting or no tool was used
            # Prefer to return the assistant's final generated content if available
            assistant_content = None
            try:
                assistant_content = last_msg.content if hasattr(last_msg, "content") else (last_msg.get("content") if isinstance(last_msg, dict) else None)
            except Exception:
                assistant_content = None
            final_reply = assistant_content or "I've processed your request, but no changes were made to the document."

        return {
            "reply": final_reply,
            "updated_long_text": final_state.get("document"),
            "is_updated": final_state.get("is_updated"),
            "change_summary": final_state.get("change_summary"),
            "error_details": final_state.get("error_log")
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

# ------------------------------------------------------------------
# 6. SET GLOBAL RESULT
# ------------------------------------------------------------------
if any(k in globals() for k in ("inputs", "input", "payload")):
    globals()["result"] = run_updater_agent()
