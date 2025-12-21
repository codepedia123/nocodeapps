# main.py - LangChain dynamic API runtime with robust JSON parsing
import os
import json
import requests
import traceback
import time
import uuid
import urllib.parse
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# LangChain imports
from langchain_core.tools import StructuredTool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# ---------------------------
# Configuration JSON
# ---------------------------
DYNAMIC_CONFIG = {
    "5": {
        "api_url": "https://ap.rhythmflow.ai/api/v1/webhooks/UZ6KJw8w1EInlLqhP1gWZ/sync",
        "api_payload_json": "%7B%22values%22%3A%7B%22A%22%3A%22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C%22B%22%3A%22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C%22C%22%3A%22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%7D%7D",
        "instructions": "'values.A' This is a Non-Required parameter, its value should be the name of the person, if provided, and can be any short text format; 'values.B' This is a Non-Required parameter, its value should be the email address, if provided, and must follow the standard email format; 'values.C' This is a Non-Required parameter, its value should be the phone number, if provided, and can be any short text format.",
        "agent_id": "1",
        "created_at": "1766317934",
        "updated_at": "1766317934",
        "when_run": "When user asks for email"
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
# Dynamic Tool Factory (Robust Version)
# ---------------------------
def create_universal_tools(config: Dict[str, Any]) -> List[StructuredTool]:
    generated_tools = []

    for tool_id, cfg in config.items():
        api_url = cfg["api_url"]
        instructions = cfg["instructions"]
        condition = cfg["when_run"]
        raw_payload_template = urllib.parse.unquote(cfg["api_payload_json"])
        
        # We use Any for tool_input to prevent Pydantic validation errors when the agent sends a string
        def make_api_call(tool_input: Any, url=api_url, tid=tool_id) -> str:
            event_id = str(uuid.uuid4())
            payload = {}

            # Parse string to dictionary if the agent sends a JSON string
            if isinstance(tool_input, str):
                try:
                    # Remove potential markdown code blocks
                    cleaned = tool_input.strip().strip('`')
                    if cleaned.startswith('json'):
                        cleaned = cleaned[4:].strip()
                    payload = json.loads(cleaned)
                except Exception as e:
                    return f"Error: Input was not valid JSON. You provided: {tool_input}. Parse Error: {str(e)}"
            else:
                payload = tool_input

            logger.log("tool.call", f"api_tool_{tid} triggered", {"event_id": event_id, "payload": payload})
            
            start = time.time()
            try:
                resp = requests.post(url, json=payload, timeout=15)
                elapsed = time.time() - start
                logger.log("http.request", f"POST to {url}", {"status_code": resp.status_code, "elapsed_s": elapsed})
                resp.raise_for_status()
                return f"Success: {resp.text}"
            except Exception as e:
                logger.log("tool.error", f"api_tool_{tid} failed", {"error": str(e)})
                return f"API Call failed: {str(e)}"

        tool_desc = (
            f"WHEN TO RUN: {condition}. "
            f"DATA MAPPING: {instructions}. "
            f"INPUT FORMAT: You MUST provide a JSON object matching this schema: {raw_payload_template}"
        )

        new_tool = StructuredTool.from_function(
            func=make_api_call,
            name=f"api_tool_{tool_id}",
            description=tool_desc
        )
        generated_tools.append(new_tool)
    
    return generated_tools

tools = create_universal_tools(DYNAMIC_CONFIG)

# ---------------------------
# ReAct Prompt
# ---------------------------
REACT_PROMPT = PromptTemplate.from_template(
    """
You are a highly efficient data synchronization assistant.

Available tools: {tool_names}
{tools}

To use a tool, follow this format exactly:

Question: the input question you must answer
Thought: I need to sync data using api_tool_X. I will construct the JSON payload.
Action: the tool name (one of: {tool_names})
Action Input: A RAW JSON OBJECT (No markdown, no backticks). Example: {{"row_id": 0, "values": {{"A": "data"}}}}
Observation: the tool result
... (repeat Thought/Action/Action Input/Observation if needed)
Thought: I have finished the task.
Final Answer: [Short confirmation to user]

RULES:
1. ONLY provide a JSON object in the 'Action Input'.
2. If data for a field is unknown, use an empty string "" unless specified otherwise.
3. Stop immediately after the Final Answer.

Question: {input}
{agent_scratchpad}
"""
)

# ---------------------------
# Core Execution
# ---------------------------
def run_agent(conversation_history, message, provider, api_key):
    logger.clear()
    logger.log("run.start", "Dynamic agent invoked", {"provider": provider})

    api_key_to_use = api_key or os.getenv("OPENAI_API_KEY")

    try:
        if provider == "groq":
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        else:
            if not api_key_to_use:
                return {"reply": "Error: No API Key provided.", "logs": logger.to_list()}
            llm = ChatOpenAI(api_key=api_key_to_use, model="gpt-4o-mini", temperature=0)
        
        agent = create_react_agent(llm, tools, REACT_PROMPT)
        executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, max_iterations=5)

        # Context construction
        history_lines = []
        for turn in conversation_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history_lines.append(f"{role}: {content}")
        
        conv_text = "\n".join(history_lines)
        combined_input = f"Conversation History:\n{conv_text}\n\nLatest User Request: {message}"

        response = executor.invoke({"input": combined_input})
        reply = response.get("output", "No response generated.")

        return {"reply": reply, "logs": logger.to_list(), "diagnostics": {"status": "success"}}

    except Exception as e:
        tb = traceback.format_exc()
        logger.log("run.error", str(e), {"traceback": tb})
        return {"reply": f"Internal Error: {str(e)}", "logs": logger.to_list()}

# ---------------------------
# FastAPI Endpoints
# ---------------------------
app = FastAPI()

@app.post("/run-agent")
async def run_endpoint(request: Request):
    try:
        body = await request.json()
        res = run_agent(
            body.get("conversation", []),
            body.get("message", ""),
            body.get("provider", "openai"),
            body.get("api_key")
        )
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Compatibility for execution environments
if "inputs" in globals():
    data = globals().get("inputs", {})
    _out = run_agent(
        data.get("conversation", []),
        data.get("message", ""),
        data.get("provider", "openai"),
        data.get("api_key", "")
    )
    globals()["result"] = _out
