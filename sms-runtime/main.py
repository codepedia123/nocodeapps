# main.py - Modern Tool Calling Agent (Dynamic Decision Making)
import os
import json
import requests
import traceback
import time
import uuid
import urllib.parse
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# LangChain modern imports
from langchain_core.tools import StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# ---------------------------
# Configuration JSON
# ---------------------------
DYNAMIC_CONFIG = {
    "5": {
        "api_url": "https://ap.rhythmflow.ai/api/v1/webhooks/UZ6KJw8w1EInlLqhP1gWZ/sync",
        "api_payload_json": "%7B%22values%22%3A%7B%22A%22%3A%22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C%22B%22%3A%22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C%22C%22%3A%22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%7D%7D",
        "instructions": "'values.A' is Name; 'values.B' is Email; 'values.C' is Phone.",
        "when_run": "ONLY run this when the user specifically provides an email or contact info to be saved/synced."
    }
}

# ---------------------------
# Logger helper
# ---------------------------
class Logger:
    def __init__(self):
        self._events: List[Dict[str, Any]] = []
    def _now(self) -> str:
        return datetime.utcnow().isoformat() + "Z"
    def log(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        entry = {"ts": self._now(), "id": str(uuid.uuid4()), "type": event_type, "message": message, "data": data or {}}
        self._events.append(entry)
    def to_list(self) -> List[Dict[str, Any]]: return self._events.copy()
    def clear(self): self._events = []

logger = Logger()

# ---------------------------
# Dynamic Tool Factory
# ---------------------------
def create_universal_tools(config: Dict[str, Any]) -> List[StructuredTool]:
    generated_tools = []

    for tool_id, cfg in config.items():
        api_url = cfg["api_url"]
        instructions = cfg["instructions"]
        condition = cfg["when_run"]
        raw_payload_template = urllib.parse.unquote(cfg["api_payload_json"])
        
        def make_api_call(tool_input: Any, url=api_url, tid=tool_id) -> str:
            event_id = str(uuid.uuid4())
            payload = {}

            # Robust parsing (handles strings or dicts)
            if isinstance(tool_input, str):
                try:
                    cleaned = tool_input.strip().strip('`').replace('json\n', '', 1)
                    payload = json.loads(cleaned)
                except:
                    return f"Error: Invalid JSON format."
            else:
                payload = tool_input

            logger.log("tool.call", f"api_tool_{tid} triggered", {"event_id": event_id, "payload": payload})
            
            try:
                resp = requests.post(url, json=payload, timeout=15)
                resp.raise_for_status()
                return f"Success: Data synced to API."
            except Exception as e:
                return f"API Call failed: {str(e)}"

        tool_desc = (
            f"CRITICAL CONDITION: {condition}. "
            f"INSTRUCTIONS: {instructions}. "
            f"REQUIRED JSON STRUCTURE: {raw_payload_template}"
        )

        new_tool = StructuredTool.from_function(
            func=make_api_call,
            name=f"sync_data_tool_{tool_id}",
            description=tool_desc
        )
        generated_tools.append(new_tool)
    
    return generated_tools

# ---------------------------
# Core Execution logic
# ---------------------------
def run_agent(conversation_history, message, provider, api_key):
    logger.clear()
    logger.log("run.start", "Agent started", {"provider": provider})

    api_key_to_use = api_key or os.getenv("OPENAI_API_KEY")
    tools = create_universal_tools(DYNAMIC_CONFIG)

    try:
        # 1. Select Model
        if provider == "groq":
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        else:
            llm = ChatOpenAI(api_key=api_key_to_use, model="gpt-4o-mini", temperature=0)
        
        # 2. Define modern Tool-Calling Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. You have tools available for data syncing. "
                       "CRITICAL RULE: Only use a tool if the user's request matches the tool's 'CRITICAL CONDITION'. "
                       "If the user is asking general questions (like 'What is ChatGPT'), do NOT use any tools. Just answer naturally."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # 3. Create Agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=False, 
            handle_parsing_errors=True
        )

        # 4. Format History
        history = []
        for turn in conversation_history:
            if turn.get("role") == "user":
                history.append(("human", turn.get("content")))
            else:
                history.append(("ai", turn.get("content")))

        # 5. Execute
        response = executor.invoke({
            "input": message,
            "chat_history": history
        })
        
        return {
            "reply": response["output"], 
            "logs": logger.to_list()
        }

    except Exception as e:
        logger.log("run.error", str(e), {"traceback": traceback.format_exc()})
        return {"reply": f"Error: {str(e)}", "logs": logger.to_list()}

# ---------------------------
# API Wrapper
# ---------------------------
app = FastAPI()

@app.post("/run-agent")
async def run_endpoint(request: Request):
    body = await request.json()
    res = run_agent(
        body.get("conversation", []),
        body.get("message", ""),
        body.get("provider", "openai"),
        body.get("api_key")
    )
    return JSONResponse(res)

if "inputs" in globals():
    data = globals().get("inputs", {})
    _out = run_agent(data.get("conversation", []), data.get("message", ""), data.get("provider", "openai"), data.get("api_key", ""))
    globals()["result"] = _out
