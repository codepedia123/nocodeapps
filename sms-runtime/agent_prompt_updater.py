import os
import json
import traceback
import re
import difflib
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

for k in (
    "long_text",
    "message",
    "conversation",
    "api_key",
    "apiKey",
    "openai_api_key",
    "key",
    "agent_id",
    "agentId",
    "agent",
    "when_run",
):
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
            except:
                pass
else:
    # Secondary env fallback for agent id if provided separately
    if "agent_id" not in data:
        env_agent = os.getenv("AGENT_ID")
        if env_agent:
            data["agent_id"] = env_agent
    if "when_run" not in data:
        env_when = os.getenv("WHEN_RUN_JSON")
        if env_when:
            data["when_run"] = env_when

long_text = data.get("long_text", "")
user_message = (
    data.get("message", "") or data.get("input_message", "") or data.get("msg", "")
)
conversation = data.get("conversation", []) or data.get("chat_history", [])
api_key_to_use = (
    data.get("api_key")
    or data.get("openai_api_key")
    or data.get("apiKey")
    or os.getenv("OPENAI_API_KEY")
)

# Helper utilities for tool management
def _parse_when_run_payload(raw_payload: Any) -> Dict[str, Any]:
    """
    Accepts a stringified JSON array or a list of objects shaped like:
      [{"tool1": ["When Run 1", "When Run 2"]}, {"tool2": ["When Run A"]}]
    Returns { "tools": [ { "tool_id": str, "when_run": list[str] } ], "error": Optional[str] }.
    """
    if raw_payload is None or raw_payload == "":
        return {"tools": [], "error": "No when_run payload provided."}

    parsed = None
    if isinstance(raw_payload, str):
        try:
            parsed = json.loads(raw_payload)
        except Exception:
            return {"tools": [], "error": "Invalid when_run JSON."}
    elif isinstance(raw_payload, list):
        parsed = raw_payload
    else:
        return {"tools": [], "error": "Unsupported when_run payload type."}

    if not isinstance(parsed, list):
        return {"tools": [], "error": "when_run payload must be a list of objects."}

    tools: List[Dict[str, Any]] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        for key, val in entry.items():
            if not key:
                continue
            if isinstance(val, list):
                cleaned = [str(x) for x in val if str(x).strip()]
            elif isinstance(val, str):
                cleaned = [v for v in [val.strip()] if v]
            else:
                cleaned = []
            tools.append({"tool_id": str(key), "when_run": cleaned})
    if not tools:
        return {"tools": [], "error": "No valid tools found in when_run payload."}
    return {"tools": tools, "error": None}


def _serialize_tools_catalog(tools_catalog: List[Dict[str, Any]]) -> str:
    """
    Converts internal tools catalog back to the expected stringified JSON array format:
      [{"tool1":["...","..."]},{"tool2":["..."]}]
    """
    output = []
    for tool in tools_catalog:
        tool_id = tool.get("tool_id")
        when = tool.get("when_run") or []
        output.append({tool_id: when})
    return json.dumps(output, ensure_ascii=True)


def _build_diff_hunks(before_text: str, after_text: str, context: int = 5) -> List[Dict[str, Any]]:
    """
    Returns unified diff hunks with context lines for client-side rendering.
    Each hunk includes a header and its lines with +/ -/ space prefixes.
    """
    before_lines = before_text.splitlines()
    after_lines = after_text.splitlines()
    diff_lines = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            lineterm="",
            n=context,
        )
    )

    hunks: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line in diff_lines:
        if line.startswith("@@"):
            if current:
                hunks.append(current)
            current = {"header": line, "lines": []}
            continue
        if line.startswith("---") or line.startswith("+++"):
            continue
        if current is None:
            current = {"header": "", "lines": []}
        current["lines"].append(line)

    if current:
        hunks.append(current)

    return hunks


def _build_word_diff_markup(before_text: str, after_text: str) -> str:
    """
    Builds a full-text, word-level diff with inline markers for additions and deletions.
    Deleted text is wrapped in <del class="diff-del">...</del>
    Added text is wrapped in <ins class="diff-add">...</ins>
    """
    def _tokenize(s: str) -> List[str]:
        return re.findall(r"\S+|\s+", s)

    a_tokens = _tokenize(before_text)
    b_tokens = _tokenize(after_text)

    matcher = difflib.SequenceMatcher(a=a_tokens, b=b_tokens)
    out: List[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out.extend(a_tokens[i1:i2])
        elif tag == "delete":
            segment = "".join(a_tokens[i1:i2])
            if segment:
                out.append(f'<del class="diff-del">{segment}</del>')
        elif tag == "insert":
            segment = "".join(b_tokens[j1:j2])
            if segment:
                out.append(f'<ins class="diff-add">{segment}</ins>')
        elif tag == "replace":
            old_seg = "".join(a_tokens[i1:i2])
            new_seg = "".join(b_tokens[j1:j2])
            if old_seg:
                out.append(f'<del class="diff-del">{old_seg}</del>')
            if new_seg:
                out.append(f'<ins class="diff-add">{new_seg}</ins>')

    return "".join(out)


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
    tools_change_summary: Optional[List[str]]
    when_run_updated: bool


@tool
def patch_document_tool(
    original_snippet: str,
    replacement_text: str,
    explanation: str,
    anchor_for_insertion: Optional[str] = None,
):
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
        "anchor_for_insertion": anchor_for_insertion,
    }


@tool
def insert_after_anchor_tool(
    anchor_snippet: str, insertion_text: str, explanation: str
):
    """
    Requests insertion of insertion_text immediately after the first occurrence of anchor_snippet.
    The executor (tool_executor) will perform the insertion on the document.
    """
    return {
        "status": "planned",
        "action": "insert_after_anchor",
        "anchor_snippet": anchor_snippet,
        "insertion_text": insertion_text,
        "explanation": explanation,
    }


@tool
def update_when_run_tool(
    tool_id: str, scenario_updates: List[Dict[str, Any]], explanation: str
):
    """
    Requests updates to a tool's when_run scenarios for the current agent payload (no remote calls).
    Provide 'scenario_updates' as a list of {'index': 1-based index to replace or append, 'text': full replacement scenario}.
    Only request this when changing when_run is essential to satisfy the user's update and stays within the provided tool arrays.
    """
    return {
        "status": "planned",
        "action": "update_when_run",
        "tool_id": tool_id,
        "scenario_updates": scenario_updates,
        "explanation": explanation,
    }


# ------------------------------------------------------------------
# 3. AGENT LOGIC UNIT
# ------------------------------------------------------------------


def run_updater_agent():
    if not api_key_to_use:
        return {"error": "Missing 'api_key' in request payload."}

    try:
        when_run_payload = data.get("when_run")
        if when_run_payload is None or (isinstance(when_run_payload, str) and not when_run_payload.strip()):
            tools_catalog = []
            tools_parse_error = None
        else:
            tool_parse_result = _parse_when_run_payload(when_run_payload)
            tools_catalog = tool_parse_result.get("tools") or []
            tools_parse_error = tool_parse_result.get("error")

        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key_to_use)
        tools = [patch_document_tool, insert_after_anchor_tool, update_when_run_tool]
        llm_with_tools = llm.bind_tools(tools)

        # 1. THE AGENT NODE
        def call_model(state: AgentState):
            tools_json = json.dumps(state.get("tools_catalog") or [], ensure_ascii=True)
            # Add explicit guidance to be conservative and prefer precise updates
            sys_msg_content = (
                "ROLE:\n"
                "You are a Global Prompt Editor for AI agents. ( Call yourself as the Agent Builder Assistant if asked. )\n\n"
                "CORE CONSTRAINT (CRITICAL):\n"
                "There exists ONLY ONE prompt. This is the GLOBAL PROMPT of the agent.\n "
                "IMPORTANT: The provided document IS the GLOBAL prompt of the agent.\n Remmeber, Any change you make in the global prompt directly changes the agent’s behavior, not actional, but what it says, what it decides, and how it thinks, not about when and how it runs a tool, if you want to update that, refer to the when_run updating, if there are tool actional scenerios related changes user have direclty said, or sounds that it's needed according to their request, indrieclty, always go for a when_run update, remember. AND ALWAYS PREFER UPDATING when_run OVER PROMPT EDITS IF THE CHANGE ( TOOL'S SCENERIO WHEN IT SHOULD RUN ) CAN BE ACHIEVED THAT WAY.\n\n"
                "This single prompt contains all behavior, rules, instructions, tone, logic, and examples.\n"
                "There are NO separate system, user, developer, or assistant prompts.\n"
                "You must NEVER split the prompt into sections or treat any part as a different role.\n\n"
                "WHAT EDITING MEANS:\n"
                "Editing means modifying the existing global prompt text itself.\n"
                "You are NOT generating a new prompt.\n"
                "You are NOT rewriting the full prompt unless explicitly asked.\n"
                "You are performing precise, minimal edits to the existing text.\n\n"
                "EDITING RULES:\n"
                "1. Default to PATCHING existing text using exact snippets.\n"
                "2. Insert new text ONLY when the content does not already exist.\n"
                "3. Never duplicate instructions that already exist in the prompt.\n"
                "4. Preserve tone, formatting, structure, and ordering unless explicitly instructed otherwise.\n"
                "5. If a request can be satisfied by modifying a single sentence, do not touch anything else.\n"
                "6. Do not add explanations, commentary, or meta text into the prompt.\n"
                "7. Do not introduce placeholders, examples, or verbosity unless the user explicitly asks.\n\n"
                "GLOBAL PROMPT OWNERSHIP:\n"
                "The provided document IS the agent.\n"
                "Any change you make directly changes the agent’s behavior.\n"
                "Treat every edit as production-critical.\n\n"
                "TOOL USAGE:\n"
                "Use patch_document_tool to replace exact text whenever possible.\n"
                "Use insert_after_anchor_tool ONLY when adding new content that does not exist.\n"
                "Anchors must be stable, short, and guaranteed to exist.\n"
                "Explanations must describe WHAT changed and WHY, not HOW.\n\n"
                "SUMMARY STYLE:\n"
                "All explanations and summaries must be short, plain, and agent-focused. Never mention the user. Only state what it now does or how it behaves.\n\n"
                f"TOOLS CATALOG (from input):\n{tools_json}\n\n"
                "WHEN_RUN UPDATE RULES:\n"
                "- MOST IMPORTANT: Update the when_run scenerio of tool/s ( ONLY WHEN THE NEED IS CLEAR ) I HAVE EXPLAINED THE RULES BELOW IN DETAIL.\nMORE BELWOW.\n"
                "- Definition: 'when_run' is the exact scenario text that decides when a specific tool should execute. If when_run is wrong, the tool will fire at the wrong time even if the global prompt text sounds correct.\n"
                "- Primary goal: Keep tool execution correct, sequential (when required), and consent-safe. Treat when_run as the execution contract for each tool.\n"
                "- Mandatory precheck (always): For every user request, first decide whether the request changes any of these: execution order, timing, triggers, prerequisites, consent requirements, conditional gating, dependencies between actions, or what data must exist before a tool runs. If YES to any, you must evaluate tool when_run alignment before making or finalizing prompt edits.\n"
                "- Tool alignment is not optional: If a request affects behavior that depends on tools, you must not consider the change complete until the relevant tools’ when_run scenarios are aligned with the requested behavior.\n"
                "- Direct request handling: If the user explicitly says to change when a tool should run, update that tool’s when_run scenarios.\n"
                "- Indirect request handling (critical): If the user asks for an edit, fix, or behavioral change without mentioning tools or when_run, you must still infer whether tools are involved and whether one or more tools’ when_run scenarios must change to fulfill the request. Do not wait for the user to tell you to update a tool.\n"
                "- When to update when_run (required conditions): Update a tool’s when_run when the current scenario would make the tool run too early, too late, automatically when it should be gated, without required user consent, without required prerequisites, out of sequence relative to other tools, or in a way that could partially or incorrectly fulfill the user’s intent.\n"
                "- When NOT to update when_run: Do not update when_run if the requested change is purely conversational or wording-only and does not change tool triggering, sequencing, prerequisites, consent, or data dependencies. Avoid defensive or speculative updates.\n"
                "- Multi-tool reasoning (required): If multiple tools are part of the requested workflow, treat them as an interdependent chain. Verify that their when_run scenarios cooperate and cannot collide. If the request changes any step in the chain, evaluate each tool and update every affected tool, not just the most obvious one.\n"
                "- If you decide that a tool’s when_run must be updated, you must always call 'update_when_run_tool' with a non-empty 'scenario_updates' array containing at least one explicit index-based replacement; explanations alone are never sufficient."
                "- Never describe an intended when_run change without providing the corresponding 'scenario_updates'; doing so is considered an invalid update."
                "- Sequential execution rule: When the user wants tools to run sequentially, encode the sequence explicitly in when_run using clear prerequisites, for example: 'run Tool B only after Tool A succeeded' and 'run Tool C only after user confirmed X'. Make sure earlier tools do not trigger later tools prematurely.\n"
                "- Consent gating rule: If a tool records, stores, inserts, sends, or persists user data (for example, inserting rows into a sheet), its when_run must explicitly require user consent when consent is part of the user’s request or the global prompt’s policy.\n"
                "- Prompt edits vs tool fixes (priority): If a behavioral request can be solved by either changing the global prompt text or correcting tool execution logic, prefer correcting tool execution logic first. Never use prompt edits as a substitute for fixing incorrect tool timing, gating, prerequisites, or sequence.\n"
                "- No unnecessary clarification: Do not ask the user to clarify tool sequencing if the intent is reasonably inferable (for example, 'book first then insert row', 'check availability before booking', 'ask consent before storing'). Infer it and update when_run accordingly.\n"
                "- Implementation constraints: Use only tools that exist in the provided tools catalog. Use the exact 'tool_id' string from the catalog. Never invent, rename, or merge tools.\n"
                "- Update method: You may update one tool or multiple tools in the same request. If multiple tools are affected, you must call 'update_when_run_tool' separately for each affected tool.\n"
                "- Scenario editing method: Prefer editing existing scenarios. Append a new scenario only when the needed behavior cannot be represented by editing an existing one.\n"
                "- Index rules: Scenario indices are 1-based. For every scenario you modify, send {'index': <n>, 'text': <full replacement scenario>} with the full final scenario text (no partial fragments). To add a scenario, use index = current_length + 1 with no gaps.\n"
                "- Scope control: Only change when_run text that is necessary to align execution with the user’s request. Avoid unrelated edits.\n"
                "- Explanation requirement: Every 'update_when_run_tool' call must include a clear, specific explanation of why the change is required to satisfy the user’s intent (for example: prevents premature execution, enforces consent, enforces sequence, enforces prerequisite).\n\n"
                f"CURRENT AGENT'S GLOBAL PROMPT DOCUMENT CONTENT:\n--- START ---\n{state['document']}\n--- END ---\n\n"
                "Answer format: If you want to call a tool, call the appropriate tool with JSON args. Otherwise reply normally.\n"
            )
            sys_msg = SystemMessage(content=sys_msg_content)
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
            tools_summary = state.get("tools_change_summary") or []
            when_run_updated = state.get("when_run_updated", False)

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

                def _append_tool_summary(text: str):
                    if text:
                        tools_summary.append(text)

                # Patch action
                if name == "patch_document_tool" or (
                    isinstance(tool_call, dict)
                    and tool_call.get("tool") == "patch_document_tool"
                ):
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
                                new_doc = (
                                    new_doc[:end_idx] + insertion + new_doc[end_idx:]
                                )
                                _append_summary(
                                    expl or "Inserted new text after provided anchor."
                                )
                                updated = True
                                err = None
                            else:
                                # fallback 2: true secondary fallback, append at end
                                new_doc = new_doc.rstrip() + "\n\n" + new.strip() + "\n"
                                _append_summary(
                                    expl
                                    or "Anchor not found, appended new section at end."
                                )
                                updated = True
                                err = None
                        else:
                            # fallback 2: true secondary fallback, append at end
                            new_doc = new_doc.rstrip() + "\n\n" + new.strip() + "\n"
                            _append_summary(
                                expl
                                or "No snippet match and no anchor provided, appended new section at end."
                            )
                            updated = True
                            err = None

                # Insert-after-anchor action
                elif name == "insert_after_anchor_tool" or (
                    isinstance(tool_call, dict)
                    and tool_call.get("tool") == "insert_after_anchor_tool"
                ):
                    anchor = args.get("anchor_snippet", "") or ""
                    insertion_text = args.get("insertion_text", "") or ""
                    expl = args.get("explanation", "") or ""

                    if insertion_text and insertion_text in new_doc:
                        _append_summary(
                            expl or "Insertion skipped because the text already exists."
                        )
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
                        new_doc = (
                            new_doc.rstrip() + "\n\n" + insertion_text.strip() + "\n"
                        )
                        _append_summary(
                            expl or "Anchor not found, appended new section at end."
                        )
                        updated = True
                        err = None

                # Update when_run action
                elif name == "update_when_run_tool" or (
                    isinstance(tool_call, dict)
                    and tool_call.get("tool") == "update_when_run_tool"
                ):
                    tool_id = str(args.get("tool_id") or "").strip()
                    scenario_updates = args.get("scenario_updates") or []
                    expl = args.get("explanation", "") or ""

                    if not tool_id:
                        err = "Missing tool_id for when_run update."
                        _append_tool_summary(expl or err)
                        continue

                    if not isinstance(scenario_updates, list) or not scenario_updates:
                        err = f"No scenario_updates provided for tool {tool_id}."
                        _append_tool_summary(expl or err)
                        continue

                    target_tool = next(
                        (t for t in tools_catalog if str(t.get("tool_id")) == tool_id),
                        None,
                    )
                    if not target_tool:
                        err = f"Tool {tool_id} not found for current agent."
                        _append_tool_summary(expl or err)
                        continue

                    current_when = list(target_tool.get("when_run") or [])
                    if not current_when and target_tool.get("raw_when_run"):
                        current_when = _split_when_run(
                            target_tool.get("raw_when_run") or ""
                        )

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
                        _append_tool_summary(expl or "No when_run changes applied.")
                        continue

                    new_when_raw = "|".join(current_when)
                    target_tool["when_run"] = current_when
                    if out_of_range_note:
                        _append_tool_summary(out_of_range_note)
                    _append_tool_summary(
                        expl or f"when_run for tool {tool_id} updated locally."
                    )
                    when_run_updated = True
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
                "tools_catalog": tools_catalog,
                "tools_change_summary": tools_summary,
                "when_run_updated": when_run_updated,
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
            "error_log": tools_parse_error,
            "tools_catalog": tools_catalog,
            "tools_change_summary": [],
            "when_run_updated": False,
        }

        final_state = app.invoke(inputs)
        original_text = long_text
        updated_text = final_state.get("document") or ""
        diff_context = 5
        diff_hunks = _build_diff_hunks(original_text, updated_text, context=diff_context)
        diff_full_marked_text = _build_word_diff_markup(original_text, updated_text)

        # ------------------------------------------------------------------
        # 5. CONSTRUCT FINAL RESPONSE (Improved Logic)
        # ------------------------------------------------------------------
        last_msg = final_state["messages"][-1] if final_state.get("messages") else None

        if final_state.get("is_updated"):
            # Scenario A: Tool worked perfectly
            summary = "; ".join(final_state.get("change_summary") or [])
            tool_summary = "; ".join(final_state.get("tools_change_summary") or [])
            combined = "; ".join([s for s in [summary, tool_summary] if s])
            if combined:
                final_reply = f"It now has these changes: {combined}".strip()
            else:
                final_reply = "It now has the requested changes."
        elif final_state.get("error_log"):
            # Scenario B: Tool was called but the snippet didn't match
            summary = "; ".join(final_state.get("change_summary") or [])
            tool_summary = "; ".join(final_state.get("tools_change_summary") or [])
            combined = "; ".join([s for s in [summary, tool_summary] if s])
            final_reply = (
                f"I tried to make the following change: '{combined}', but hit an issue: {final_state.get('error_log')}. "
                "For document edits, ensure the snippet or anchor (for example '# Notes:' or '# Task:') is present. "
                "For tool updates, verify the tool_id and 1-based when_run indices match the catalog."
            )
        else:
            # Scenario C: Chatting or no tool was used
            # Prefer to return the assistant's final generated content if available
            assistant_content = None
            try:
                assistant_content = (
                    last_msg.content
                    if hasattr(last_msg, "content")
                    else (
                        last_msg.get("content") if isinstance(last_msg, dict) else None
                    )
                )
            except Exception:
                assistant_content = None
            final_reply = (
                assistant_content
                or "I've processed your request, but no changes were made to the document."
            )

        return {
            "reply": final_reply,
            "updated_long_text": final_state.get("document"),
            "is_updated": final_state.get("is_updated"),
            "change_summary": final_state.get("change_summary"),
            "tools_change_summary": final_state.get("tools_change_summary"),
            "when_run_updated": final_state.get("when_run_updated"),
            "current_when_run_json": _serialize_tools_catalog(
                final_state.get("tools_catalog") or []
            ),
            "diff_context": diff_context,
            "diff_hunks": diff_hunks,
            "diff_full_marked_text": diff_full_marked_text,
            "error_details": final_state.get("error_log"),
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


# ------------------------------------------------------------------
# 6. SET GLOBAL RESULT
# ------------------------------------------------------------------
if any(k in globals() for k in ("inputs", "input", "payload")):
    globals()["result"] = run_updater_agent()
