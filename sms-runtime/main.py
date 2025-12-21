# main.py - LangChain dynamic API runtime with full step-by-step logging
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
    "3": {
        "api_url": "https://ap.rhythmflow.ai/api/v1/webhooks/I8pJYgOFUaIqx5SfudeHn/sync",
        "api_payload_json": "%7B %22row_id%22%3A 0%2C %22values%22%3A %7B %22A%22%3A %22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C %22B%22%3A %22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C %22C%22%3A %22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C %22D%22%3A %22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C %22E%22%3A %22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C %22F%22%3A %22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C %22G%22%3A %22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C %22H%22%3A %22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C %22I%22%3A %22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C %22J%22%3A %22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C %22K%22%3A %22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22 %7D %7D",
        "instructions": "'row_id' This is a Required parameter, its value should be the zero-based index for the row number you intend to update; 'values.A' is timestamp; 'values.B' is phone; 'values.C' is name; 'values.D' is email; 'values.E' is street address; 'values.F' is city; 'values.G' is state; 'values.H' is ZIP; 'values.I' is age; 'values.J' is DOB; 'values.K' is 'Qualified + Transferred' status.",
        "when_run": "Run this whenever user asks for email or provides contact information to be synced/updated."
    },
    "4": {
        "api_url": "https://ap.rhythmflow.ai/api/v1/webhooks/DDBxxS1Ja2b1PgrHE1R3w/sync",
        "api_payload_json": "%7B%22values%22%3A%7B%22A%22%3A%22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C%22B%22%3A%22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C%22C%22%3A%22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C%22D%22%3A%22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C%22E%22%3A%22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C%22F%22%3A%22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C%22G%22%3A%22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C%22H%22%3A%22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C%22I%22%3A%22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C%22J%22%3A%22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%2C%22K%22%3A%22%7B%7BREPLACE_WITH_ACTUAL_VALUE%7D%7D%22%7D%7D",
        "instructions": "'A' is Timestamp; 'B' is Phone; 'C' is Name; 'D' is Email; 'E' is Address; 'F' is City; 'G' is State; 'H' is ZIP; 'I' is Age; 'J' is DOB; 'K' is Qualified + Transferred status.",
        "when_run": "Run this whenever user provides contact details or requests a status update for a lead."
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
        # Decode the payload template for reference (though the LLM creates the actual payload)
        raw_payload_str = urllib.parse.unquote(cfg["api_payload_json"])
        
        def make_api_call(payload: Dict[str, Any], url=api_url, tid=tool_id) -> str:
            event_id = str(uuid.uuid4())
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

        # Create StructuredTool with dynamic description
        # We tell the LLM exactly how to format the JSON payload via the description
        tool_desc = (
            f"USE CASE: {condition}. "
            f"INSTRUCTIONS: {instructions}. "
            f"The input must be a valid JSON object following the structure: {raw_payload_str}"
        )

        new_tool = StructuredTool.from_function(
            func=make_api_call,
            name=f"api_tool_{tool_id}",
            description=tool_desc
        )
        generated_tools.append(new_tool)
    
    return generated_tools

# Register tools
tools = create_universal_tools(DYNAMIC_CONFIG)

# ---------------------------
# ReAct Prompt
# ---------------------------
REACT_PROMPT = PromptTemplate.from_template(
    """
You are an expert data assistant. You have access to specific API tools to sync user data.

Available tools: {tool_names}
{tools}

To use a tool, you MUST use this exact format:

Thought: I need to use api_tool_X because [reason]. I will curate the payload based on the tool's instructions.
Action: the tool name (one of: {tool_names})
Action Input: A valid JSON object containing the parameters required by the tool instructions.
Observation: the tool result
... (repeat Thought/Action/Action Input/Observation if needed)
Thought: I have successfully synced the data.
Final Answer: [Concise confirmation to the user]

RULES:
1. Only call a tool if the user's request matches the 'USE CASE' in the tool description.
2. Curate the 'Action Input' JSON carefully, mapping user info to the keys (like A, B, C, row_id) as defined in the instructions.
3. If information is missing, use an empty string or null for non-required fields.

Question: {input}
{agent_scratchpad}
"""
)

# ---------------------------
# Core Logic & Execution
# ---------------------------
def _mask_key(k: Optional[str]) -> str:
    return k[:6] + "..." + k[-4:] if k and len(k) > 10 else "NO_KEY"

def make_executor(agent_obj, tools_list):
    return AgentExecutor(agent=agent_obj, tools=tools_list, handle_parsing_errors=True, max_iterations=6, verbose=False)

def extract_response_text(resp) -> str:
    if isinstance(resp, dict):
        return resp.get("output", resp.get("result", str(resp)))
    return str(resp)

def run_agent(conversation_history, message, provider, api_key):
    logger.clear()
    logger.log("run.start", "Starting dynamic agent", {"provider": provider})

    api_key_to_use = api_key or os.getenv("OPENAI_API_KEY")

    try:
        if provider == "groq":
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        else:
            llm = ChatOpenAI(api_key=api_key_to_use, model="gpt-4o-mini", temperature=0)
        
        agent = create_react_agent(llm, tools, REACT_PROMPT)
        executor = make_executor(agent, tools)

        # Context building
        conv_text = "\n".join([f"{i.get('role')}: {i.get('content')}" for i in conversation_history if isinstance(i, dict)])
        combined_input = f"History:\n{conv_text}\n\nUser: {message}"

        resp_raw = executor.invoke({"input": combined_input})
        resp_text = extract_response_text(resp_raw)

        return {"reply": resp_text, "logs": logger.to_list(), "diagnostics": {"status": "completed"}}

    except Exception as e:
        logger.log("run.error", str(e), {"traceback": traceback.format_exc()})
        return {"reply": f"Error: {str(e)}", "logs": logger.to_list()}

# ---------------------------
# FastAPI
# ---------------------------
app = FastAPI()

@app.post("/run-agent")
async def run(request: Request):
    body = await request.json()
    result = run_agent(
        body.get("conversation", []),
        body.get("message", ""),
        body.get("provider", "openai"),
        body.get("api_key")
    )
    return JSONResponse(result)

if "inputs" in globals():
    data = globals().get("inputs", {})
    _res = run_agent(data.get("conversation", []), data.get("message", ""), data.get("provider", "openai"), data.get("api_key", ""))
    globals()["result"] = _res
