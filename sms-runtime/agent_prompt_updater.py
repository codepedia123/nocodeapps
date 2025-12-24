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
# app.py injects 'inputs' as a global variable
data = globals().get("inputs", {})

# Extract specific fields from the payload
long_text = data.get("long_text", "")
user_message = data.get("message", "")
conversation = data.get("conversation", [])

# Extract API Key from the 'api_key' field in the request, fallback to environment
api_key_to_use = data.get("api_key") or os.getenv("OPENAI_API_KEY")

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

        # Format History
        history = []
        for turn in conversation:
            role = "human" if turn.get("role") == "user" else "ai"
            history.append((role, turn.get("content", turn.get("message", ""))))

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
if "inputs" in globals():
    globals()["result"] = run_updater_agent()
