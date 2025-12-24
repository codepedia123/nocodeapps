# agent_prompt_updater.py
import os
import json
import traceback
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ------------------------------------------------------------------
# 1. SETUP INPUTS (Patterned after main.py logic)
# ------------------------------------------------------------------
# app.py injects 'inputs' as a global variable when run via /int.
# Be robust: accept inputs from globals()["inputs"], globals()["input"], globals()["payload"],
# or direct globals that may contain the expected keys.
_raw_globals = globals()

# Merge available input sources into a single dict called data
data: Dict[str, Any] = {}

# Priority ordering: explicit 'inputs', then 'input', then 'payload'
for key in ("inputs", "input", "payload"):
    candidate = _raw_globals.get(key)
    if isinstance(candidate, dict) and candidate:
        data.update(candidate)

# If still empty, try to pick up top-level keys that might have been set directly
# This helps when a calling environment sets variables directly in globals
for k in ("long_text", "message", "conversation", "api_key", "apiKey", "openai_api_key", "key"):
    if k in _raw_globals and _raw_globals.get(k) not in (None, "", {}):
        # do not overwrite existing keys already provided
        if k not in data:
            data[k] = _raw_globals.get(k)

# As a last fallback, attempt to parse a raw JSON string from common env vars if present
# This can help when a wrapper exports the entire request body as an env var
if not data:
    for env_key in ("REQUEST_BODY", "RAW_BODY", "INPUT_JSON"):
        raw = os.getenv(env_key)
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    data.update(parsed)
                    break
            except Exception:
                # ignore parse errors
                pass

# Finally ensure data is a dict
if not isinstance(data, dict):
    data = {}

# Extract specific fields from the payload
long_text = data.get("long_text", "")
user_message = data.get("message", "") or data.get("input_message", "") or data.get("msg", "")
conversation = data.get("conversation", []) or data.get("chat_history", [])

# Extract API Key from multiple possible fields in the request, fallback to environment
api_key_to_use = (
    data.get("api_key")
    or data.get("openai_api_key")
    or data.get("apiKey")
    or data.get("key")
    or os.getenv("OPENAI_API_KEY")
)

# ------------------------------------------------------------------
# 2. DEFINE THE UPDATE TOOL
# ------------------------------------------------------------------
class UpdateTextSchema(BaseModel):
    updated_paragraph: str = Field(description="The complete, full version of the text with all requested changes applied.")
    explanation: str = Field(description="A brief summary of what was changed.")

# State to capture tool output
state_update = {"new_text": None, "explanation": None}

def update_prompt_tool_func(updated_paragraph: str, explanation: str) -> str:
    """Rewrites or modifies the main long text content. Use this when the user asks for changes."""
    state_update["new_text"] = updated_paragraph
    state_update["explanation"] = explanation
    return f"Success: Text updated. Change: {explanation}"

# ------------------------------------------------------------------
# 3. AGENT CORE LOGIC
# ------------------------------------------------------------------
def run_updater_agent():
    if not api_key_to_use:
        return {"error": "Missing 'api_key' in request payload and no environment fallback found."}

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key_to_use)

        tools = [
            StructuredTool.from_function(
                func=update_prompt_tool_func,
                name="update_prompt_tool",
                description="Rewrites or modifies the provided text. Use for tone changes, grammar, or formatting.",
                args_schema=UpdateTextSchema
            )
        ]

        system_prompt = (
            "You are a professional Editor.\n"
            "CURRENT TEXT:\n"
            "--- START ---\n"
            f"{long_text}\n"
            "--- END ---\n\n"
            "RULES:\n"
            "1. Use 'update_prompt_tool' ONLY if the user wants to change the text above.\n"
            "2. Always provide the FULL updated text in the tool arguments.\n"
            "3. If the user is just chatting, answer normally."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Format History tolerantly: accept 'content' or 'message' keys and common role names
        history = []
        if isinstance(conversation, list):
            for turn in conversation:
                if not isinstance(turn, dict):
                    continue
                role_raw = (turn.get("role") or "").lower().strip()
                content = turn.get("content", turn.get("message", ""))
                if role_raw == "user":
                    history.append(("human", content))
                else:
                    # treat assistant, ai, system, bot as ai unless explicitly 'user'
                    history.append(("ai", content))

        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

        response = executor.invoke({
            "input": user_message,
            "chat_history": history
        })

        # Construct final payload
        return {
            "reply": response.get("output", ""),
            "updated_long_text": state_update["new_text"] if state_update["new_text"] else long_text,
            "is_updated": state_update["new_text"] is not None,
            "change_summary": state_update["explanation"]
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

# ------------------------------------------------------------------
# 4. SET GLOBAL RESULT (Required by app.py)
# ------------------------------------------------------------------
if "inputs" in globals() or "input" in globals() or "payload" in globals():
    globals()["result"] = run_updater_agent()
