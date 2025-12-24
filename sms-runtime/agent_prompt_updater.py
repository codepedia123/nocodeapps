# agent_prompt_updater.py
import os
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ------------------------------------------------------------------
# 1. SETUP INPUTS (Injected by app.py /int endpoint)
# ------------------------------------------------------------------
# app.py merges the JSON body into a dictionary called 'inputs'
data = locals().get("inputs", {})

# Extract specific fields from the payload
long_text = data.get("long_text", "")
user_message = data.get("message", "")
conversation = data.get("conversation", [])
# Extract API Key from the 'api_key' object/field in the request
api_key = data.get("api_key")

# ------------------------------------------------------------------
# 2. DEFINE THE UPDATE TOOL
# ------------------------------------------------------------------
class UpdateTextSchema(BaseModel):
    updated_paragraph: str = Field(description="The complete, full version of the text with all requested changes applied.")
    explanation: str = Field(description="A brief summary of what was changed.")

# This variable captures tool output to return to the main runtime
state_update = {"new_text": None, "explanation": None}

def update_prompt_content(updated_paragraph: str, explanation: str) -> str:
    """Use this tool only when the user asks to modify, rewrite, or fix the text paragraph provided."""
    state_update["new_text"] = updated_paragraph
    state_update["explanation"] = explanation
    return f"Success: The text has been updated. Summary: {explanation}"

# ------------------------------------------------------------------
# 3. CONSTRUCT THE AGENT
# ------------------------------------------------------------------
def run_agent():
    # Validation: Ensure API Key is present in the request
    if not api_key:
        return {"error": "Missing 'api_key' in request payload."}

    # Initialize LLM with the provided key
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)

    tools = [
        StructuredTool.from_function(
            func=update_prompt_content,
            name="update_prompt_tool",
            description="Rewrites or modifies the main long text content.",
            args_schema=UpdateTextSchema
        )
    ]

    system_prompt = (
        "You are a professional Prompt Engineer and Editor.\n"
        "You are working on the following TEXT:\n"
        "--- START TEXT ---\n"
        f"{long_text}\n"
        "--- END TEXT ---\n\n"
        "YOUR TASKS:\n"
        "1. If the user asks to change the text (e.g., 'make it pirate tone', 'fix grammar'), "
        "you MUST use the 'update_prompt_tool' and provide the FULL updated text.\n"
        "2. If the user asks a general question, answer normally.\n"
        "3. Always return the full content when updating."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Convert conversation list to LangChain history format
    history = []
    for turn in conversation:
        role = "human" if turn.get("role") == "user" else "ai"
        history.append((role, turn.get("content", "")))

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    try:
        response = executor.invoke({
            "input": user_message,
            "chat_history": history
        })

        # ------------------------------------------------------------------
        # 4. PREPARE THE RESULT FOR APP.PY
        # ------------------------------------------------------------------
        # Assigning to 'result' is crucial for app.py to return the data
        return {
            "reply": response["output"],
            "updated_long_text": state_update["new_text"] if state_update["new_text"] else long_text,
            "is_updated": state_update["new_text"] is not None,
            "change_summary": state_update["explanation"]
        }

    except Exception as e:
        return {"error": str(e)}

# Execute the logic and store in 'result' for the app.py response
result = run_agent()
