import os
import json
import traceback
import re
from typing import List, Dict, Any, Optional, Annotated, TypedDict
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

for k in ("long_text", "message", "conversation", "api_key", "apiKey", "openai_api_key", "key"):
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

long_text = data.get("long_text", "")
user_message = data.get("message", "") or data.get("input_message", "") or data.get("msg", "")
conversation = data.get("conversation", []) or data.get("chat_history", [])
api_key_to_use = (data.get("api_key") or data.get("openai_api_key") or data.get("apiKey") or os.getenv("OPENAI_API_KEY"))

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

# ------------------------------------------------------------------
# 3. AGENT LOGIC UNIT
# ------------------------------------------------------------------

def run_updater_agent():
    if not api_key_to_use:
        return {"error": "Missing 'api_key' in request payload."}

    try:
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key_to_use)
        tools = [patch_document_tool, insert_after_anchor_tool]
        llm_with_tools = llm.bind_tools(tools)

        # 1. THE AGENT NODE
        def call_model(state: AgentState):
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
                f"CURRENT DOCUMENT CONTENT:\n--- START ---\n{state['document']}\n--- END ---\n\n"
                "Answer format: If you want to call a tool, call the appropriate tool with JSON args. Otherwise reply normally.\n"
            ))
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

                else:
                    # Unknown tool name: capture for diagnostics
                    err = f"Tool call with unknown tool name: {name}"
                    summary = summary or None

            return {"document": new_doc, "change_summary": summary, "is_updated": updated, "error_log": err}

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
            "error_log": None
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
                f"I tried to make the following change: '{summary}', "
                f"but I couldn't find the exact matching text in the document to replace or the anchor was missing. "
                "Please ensure the text or an anchor (for example '# Notes:' or '# Task:') is present exactly as you'd like it changed."
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
