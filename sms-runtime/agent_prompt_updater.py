import os
import json
import traceback
from typing import List, Dict, Any, Optional, Annotated, TypedDict
from pydantic import BaseModel, Field

# LangChain & LangGraph Imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

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
# 2. DEFINE THE STATE & TOOLS (The Official LangGraph Way)
# ------------------------------------------------------------------

class AgentState(TypedDict):
    """The official way to track state across the agent's 'thought' process."""
    messages: Annotated[List[BaseMessage], "The conversation messages"]
    document: str  # This holds the long text without sending it back/forth in ToolMessages
    change_summary: Optional[str]
    is_updated: bool

@tool
def patch_document_tool(original_snippet: str, replacement_text: str, explanation: str):
    """
    Updates the document by replacing a specific snippet with new text.
    Use this to edit the text efficiently without rewriting the whole thing.
    """
    # This function is essentially a 'schema' for the LLM. 
    # The actual logic happens inside the Graph State update.
    return f"Success: Modified snippet. Change: {explanation}"

# ------------------------------------------------------------------
# 3. AGENT LOGIC UNIT
# ------------------------------------------------------------------

def run_updater_agent():
    if not api_key_to_use:
        return {"error": "Missing 'api_key'."}

    try:
        # Initialize LLM with Tools
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key_to_use)
        tools = [patch_document_tool]
        llm_with_tools = llm.bind_tools(tools)

        # 1. THE AGENT NODE
        def call_model(state: AgentState):
            sys_msg = SystemMessage(content=(
                "You are a professional Editor.\n"
                f"CURRENT DOCUMENT CONTENT:\n--- START ---\n{state['document']}\n--- END ---\n\n"
                "RULES:\n"
                "1. To edit, use 'patch_document_tool'. Provide the EXACT original snippet to find.\n"
                "2. Do NOT rewrite the whole document. Only provide the new snippet in the tool.\n"
                "3. If the user is just chatting, reply normally."
            ))
            response = llm_with_tools.invoke([sys_msg] + state["messages"])
            return {"messages": [response]}

        # 2. THE TOOL EXECUTION NODE (Updates state directly)
        def tool_executor(state: AgentState):
            last_msg = state["messages"][-1]
            new_doc = state["document"]
            summary = state["change_summary"]
            updated = state["is_updated"]

            for tool_call in last_msg.tool_calls:
                args = tool_call["args"]
                old, new, expl = args["original_snippet"], args["replacement_text"], args["explanation"]
                if old in new_doc:
                    new_doc = new_doc.replace(old, new, 1) # Only replace first occurrence
                    summary = expl
                    updated = True
            
            # Return the updated document to the graph state
            return {"document": new_doc, "change_summary": summary, "is_updated": updated}

        # 3. BUILD THE GRAPH
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_executor)
        
        workflow.set_entry_point("agent")
        
        # Conditional logic: If LLM called a tool, go to tools node, else end
        def should_continue(state: AgentState):
            if state["messages"][-1].tool_calls:
                return "tools"
            return END

        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", END) # End after one edit for safety/efficiency

        app = workflow.compile()

        # 4. EXECUTE
        # Convert history
        history = []
        for turn in conversation:
            content = turn.get("content", turn.get("message", ""))
            role = (turn.get("role") or "").lower()
            if "user" in role: history.append(HumanMessage(content=content))
            else: history.append(BaseMessage(content=content, type="ai"))

        inputs = {
            "messages": history + [HumanMessage(content=user_message)],
            "document": long_text,
            "change_summary": None,
            "is_updated": False
        }

        final_state = app.invoke(inputs)

        return {
            "reply": final_state["messages"][-1].content,
            "updated_long_text": final_state["document"],
            "is_updated": final_state["is_updated"],
            "change_summary": final_state["change_summary"]
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

# ------------------------------------------------------------------
# 4. SET GLOBAL RESULT
# ------------------------------------------------------------------
if any(k in globals() for k in ("inputs", "input", "payload")):
    globals()["result"] = run_updater_agent()
