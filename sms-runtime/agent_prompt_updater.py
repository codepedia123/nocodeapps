import os
import json
import uuid
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# LangChain modern imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ---------------------------
# 1. Structured Tool Schema
# ---------------------------
class UpdatePromptSchema(BaseModel):
    updated_paragraph: str = Field(
        description="The full, complete version of the paragraph with all requested changes applied."
    )
    change_summary: str = Field(
        description="A brief description of what was changed (e.g., 'Added professional tone' or 'Fixed grammar')."
    )

# ---------------------------
# 2. Tool Logic
# ---------------------------
def update_text_content(updated_paragraph: str, change_summary: str) -> str:
    """
    Use this tool ONLY when the user asks to modify, update, add to, 
    or delete sections of the prompt text/paragraph.
    Returns a structured update for the UI and Database.
    """
    # This return value is what the LLM sees. 
    # We include a special flag 'UPDATE_SIGNAL' so the wrapper knows to sync DB/UI.
    result = {
        "status": "UPDATE_SIGNAL",
        "new_text": updated_paragraph,
        "summary": change_summary,
        "timestamp": datetime.utcnow().isoformat()
    }
    return json.dumps(result)

# ---------------------------
# 3. Agent Execution Logic
# ---------------------------
def run_prompt_builder_agent(
    current_long_text: str, 
    message: str, 
    conversation_history: List[Dict[str, str]],
    openai_api_key: Optional[str] = None
):
    """
    Logic for the Prompt Builder Agent.
    """
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    
    # Define the tool
    tools = [
        StructuredTool.from_function(
            func=update_text_content,
            name="update_content_tool",
            description="Update the main prompt text. Input must be the ENTIRE revised text.",
            args_schema=UpdatePromptSchema
        )
    ]

    try:
        # 1. Setup LLM (Higher temperature for creativity, or 0 for precision)
        llm = ChatOpenAI(api_key=api_key, model="gpt-4o", temperature=0)

        # 2. Construct the System Prompt
        # We inject the 'current_long_text' directly into the system instructions
        system_message = (
            "You are an expert Prompt Engineer and Editor.\n"
            "Your goal is to help the user refine a specific piece of text (the Prompt).\n\n"
            "CURRENT PROMPT TEXT:\n"
            "-------------------\n"
            f"{current_long_text}\n"
            "-------------------\n\n"
            "GUIDELINES:\n"
            "1. If the user asks a general question or seeks advice, answer naturally without updating the text.\n"
            "2. If the user asks to change, rewrite, or improve the text, call the 'update_content_tool' "
            "with the ENTIRE newly written text.\n"
            "3. Always maintain the original intent unless asked otherwise.\n"
            "4. When calling the tool, do not truncate the text. Provide the full version."
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # 3. Create Agent and Executor
        agent = create_tool_calling_agent(llm, tools, prompt_template)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True # Crucial for detecting tool calls
        )

        # 4. Format History
        formatted_history = []
        for turn in conversation_history:
            role = "human" if turn.get("role") == "user" else "ai"
            formatted_history.append((role, turn.get("content", "")))

        # 5. Invoke
        result = executor.invoke({
            "input": message,
            "chat_history": formatted_history
        })

        # 6. Post-Process Output
        # We check if a tool was called to identify if an update happened.
        agent_reply = result.get("output", "")
        updated_text_block = None
        
        # Look through intermediate steps for our specific tool call
        for action, observation in result.get("intermediate_steps", []):
            if action.tool == "update_content_tool":
                try:
                    tool_data = json.loads(observation)
                    if tool_data.get("status") == "UPDATE_SIGNAL":
                        updated_text_block = tool_data.get("new_text")
                except:
                    pass

        return {
            "reply": agent_reply,
            "updated_long_text": updated_text_block, # This will be None if no update occurred
            "is_updated": updated_text_block is not None
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"reply": f"Error: {str(e)}", "updated_long_text": None, "is_updated": False}

# ---------------------------
# 4. FastAPI Wrapper
# ---------------------------
app = FastAPI()

@app.post("/refine-prompt")
async def refine_prompt_endpoint(request: Request):
    """
    Expected JSON:
    {
        "long_text": "The existing paragraph content...",
        "message": "Can you make this sound more like a pirate?",
        "conversation": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    }
    """
    body = await request.json()
    
    long_text = body.get("long_text", "")
    message = body.get("message", "")
    conversation = body.get("conversation", [])
    
    # Run the agent
    response_data = run_prompt_builder_agent(long_text, message, conversation)
    
    # Logic for your DB Update:
    if response_data["is_updated"]:
        new_content = response_data["updated_long_text"]
        # DB.save_prompt(prompt_id, new_content) <--- Call your DB logic here
        print(f"DEBUG: Updating Database with: {new_content[:50]}...")

    return JSONResponse(response_data)
