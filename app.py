from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import sqlite3
import uuid
import json
import datetime
import threading
import requests

DB_PATH = "activepieces_demo.db"

app = FastAPI(title="Activepieces-like demo API")

def now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS flows (
                   id TEXT PRIMARY KEY,
                   name TEXT,
                   version INTEGER,
                   data TEXT,
                   created_at TEXT,
                   updated_at TEXT
                 )""")
    c.execute("""CREATE TABLE IF NOT EXISTS templates (
                   id TEXT PRIMARY KEY,
                   name TEXT,
                   data TEXT,
                   created_at TEXT
                 )""")
    c.execute("""CREATE TABLE IF NOT EXISTS flow_runs (
                   id TEXT PRIMARY KEY,
                   flow_id TEXT,
                   status TEXT,
                   input TEXT,
                   output TEXT,
                   logs TEXT,
                   created_at TEXT,
                   started_at TEXT,
                   finished_at TEXT
                 )""")
    conn.commit()
    conn.close()

init_db()

def db_execute(query, params=(), fetch=False):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(query, params)
    if fetch:
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    else:
        conn.commit()
        conn.close()
        return None

def get_flow(flow_id):
    rows = db_execute("SELECT * FROM flows WHERE id = ?", (flow_id,), fetch=True)
    if not rows:
        return None
    row = rows[0]
    row["data"] = json.loads(row["data"]) if row["data"] else {}
    return row

class CreateFlowIn(BaseModel):
    name: str = "Unnamed Flow"
    definition: dict = {}

@app.get("/", response_class=HTMLResponse)
async def admin_root():
    html = """
    <html>
      <head><title>Activepieces demo admin</title></head>
      <body>
        <h2>Activepieces demo API running</h2>
        <p>Use /v1/flows, /v1/flow-templates, /v1/flow-runs</p>
      </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.post("/v1/flows")
async def create_flow(payload: CreateFlowIn):
    flow_id = str(uuid.uuid4())
    now = now_iso()
    initial = {
        "trigger": payload.definition.get("trigger", {}),
        "steps": payload.definition.get("steps", []),
        "meta": payload.definition.get("meta", {})
    }
    db_execute("INSERT INTO flows (id, name, version, data, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
               (flow_id, payload.name, 1, json.dumps(initial), now, now))
    return {"id": flow_id, "name": payload.name, "version": 1, "data": initial, "created_at": now}

@app.get("/v1/flows")
async def list_flows(page: int = 1, size: int = 20):
    offset = (page - 1) * size
    rows = db_execute("SELECT * FROM flows ORDER BY created_at DESC LIMIT ? OFFSET ?", (size, offset), fetch=True)
    total_rows = db_execute("SELECT COUNT(1) as cnt FROM flows", fetch=True)
    total = total_rows[0]["cnt"] if total_rows else 0
    for r in rows:
        r["data"] = json.loads(r["data"]) if r["data"] else {}
    return {"total": total, "page": page, "size": size, "items": rows}

@app.get("/v1/flows/{flow_id}")
async def get_flow_endpoint(flow_id: str):
    f = get_flow(flow_id)
    if not f:
        raise HTTPException(status_code=404, detail="Flow not found")
    return f

class UpdateOperation(BaseModel):
    type: str
    request: dict = {}

@app.post("/v1/flows/{flow_id}")
async def update_flow_endpoint(flow_id: str, op: UpdateOperation):
    f = get_flow(flow_id)
    if not f:
        raise HTTPException(status_code=404, detail="Flow not found")
    data = f["data"]
    typ = op.type
    req = op.request or {}
    # Supported operations:
    # replace_definition -> { definition: {...} }
    # update_name -> { name: "new name" }
    # add_action -> { action: {...}, position: optional index (append if missing) }
    # remove_action -> { index: int }
    # set_trigger -> { trigger: {...} }
    # publish -> { published: true/false } (stored in meta.published)
    if typ == "replace_definition":
        data = req.get("definition", {})
        # normalize to trigger + steps + meta
        newdata = {
            "trigger": data.get("trigger", {}),
            "steps": data.get("steps", []),
            "meta": data.get("meta", {})
        }
        data = newdata
    elif typ == "update_name":
        new_name = req.get("name")
        if new_name:
            db_execute("UPDATE flows SET name = ?, updated_at = ? WHERE id = ?", (new_name, now_iso(), flow_id))
    elif typ == "add_action":
        action = req.get("action")
        pos = req.get("position")
        if not action:
            raise HTTPException(status_code=400, detail="action is required")
        steps = data.get("steps", [])
        if pos is None or pos < 0 or pos > len(steps):
            steps.append(action)
        else:
            steps.insert(pos, action)
        data["steps"] = steps
    elif typ == "remove_action":
        idx = req.get("index")
        if idx is None:
            raise HTTPException(status_code=400, detail="index is required")
        steps = data.get("steps", [])
        if idx < 0 or idx >= len(steps):
            raise HTTPException(status_code=400, detail="index out of range")
        steps.pop(idx)
        data["steps"] = steps
    elif typ == "set_trigger":
        trigger = req.get("trigger")
        if not trigger:
            raise HTTPException(status_code=400, detail="trigger is required")
        data["trigger"] = trigger
    elif typ == "publish":
        meta = data.get("meta", {})
        meta["published"] = bool(req.get("published", True))
        data["meta"] = meta
    elif typ == "set_definition_field":
        # sets any arbitrary path like request: { path: ["meta","author"], value: "me" }
        path = req.get("path")
        value = req.get("value")
        if not isinstance(path, list):
            raise HTTPException(status_code=400, detail="path must be a list")
        node = data
        for p in path[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[path[-1]] = value
    else:
        raise HTTPException(status_code=400, detail=f"Unknown operation type: {typ}")
    # increment version and update store
    new_version = f.get("version", 1) + 1
    now = now_iso()
    db_execute("UPDATE flows SET version = ?, data = ?, updated_at = ? WHERE id = ?",
               (new_version, json.dumps(data), now, flow_id))
    return {"id": flow_id, "version": new_version, "data": data, "updated_at": now}

class CreateTemplateIn(BaseModel):
    name: str
    definition: dict = {}

@app.post("/v1/flow-templates")
async def create_flow_template(payload: CreateTemplateIn):
    tid = str(uuid.uuid4())
    now = now_iso()
    db_execute("INSERT INTO templates (id, name, data, created_at) VALUES (?, ?, ?, ?)",
               (tid, payload.name, json.dumps(payload.definition), now))
    return {"id": tid, "name": payload.name, "data": payload.definition, "created_at": now}

@app.get("/v1/flow-runs")
async def list_flow_runs(flow_id: str = None, page: int = 1, size: int = 20):
    offset = (page - 1) * size
    if flow_id:
        rows = db_execute("SELECT * FROM flow_runs WHERE flow_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                          (flow_id, size, offset), fetch=True)
        total_rows = db_execute("SELECT COUNT(1) as cnt FROM flow_runs WHERE flow_id = ?", (flow_id,), fetch=True)
        total = total_rows[0]["cnt"] if total_rows else 0
    else:
        rows = db_execute("SELECT * FROM flow_runs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                          (size, offset), fetch=True)
        total_rows = db_execute("SELECT COUNT(1) as cnt FROM flow_runs", fetch=True)
        total = total_rows[0]["cnt"] if total_rows else 0
    for r in rows:
        r["input"] = json.loads(r["input"]) if r["input"] else {}
        r["output"] = json.loads(r["output"]) if r["output"] else {}
        r["logs"] = json.loads(r["logs"]) if r["logs"] else []
    return {"total": total, "page": page, "size": size, "items": rows}

@app.get("/v1/flow-runs/{run_id}")
async def get_flow_run(run_id: str):
    rows = db_execute("SELECT * FROM flow_runs WHERE id = ?", (run_id,), fetch=True)
    if not rows:
        raise HTTPException(status_code=404, detail="Run not found")
    r = rows[0]
    r["input"] = json.loads(r["input"]) if r["input"] else {}
    r["output"] = json.loads(r["output"]) if r["output"] else {}
    r["logs"] = json.loads(r["logs"]) if r["logs"] else []
    return r

# convenience endpoint to start a run immediately
@app.post("/v1/flows/{flow_id}/run")
async def run_flow_endpoint(flow_id: str, request: Request, background_tasks: BackgroundTasks):
    f = get_flow(flow_id)
    if not f:
        raise HTTPException(status_code=404, detail="Flow not found")
    body = {}
    try:
        body = await request.json()
    except Exception:
        body = {}
    run_id = str(uuid.uuid4())
    created = now_iso()
    db_execute("INSERT INTO flow_runs (id, flow_id, status, input, output, logs, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
               (run_id, flow_id, "pending", json.dumps(body), json.dumps({}), json.dumps([]), created))
    # run in background thread
    background_tasks.add_task(execute_flow, flow_id, run_id, body)
    return {"run_id": run_id, "status": "started", "created_at": created}

# generic POST to /v1/flow-runs to start runs by flow_id and input
class FlowRunStartIn(BaseModel):
    flow_id: str
    input: dict = {}

@app.post("/v1/flow-runs")
async def start_run_direct(payload: FlowRunStartIn, background_tasks: BackgroundTasks):
    f = get_flow(payload.flow_id)
    if not f:
        raise HTTPException(status_code=404, detail="Flow not found")
    run_id = str(uuid.uuid4())
    created = now_iso()
    db_execute("INSERT INTO flow_runs (id, flow_id, status, input, output, logs, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
               (run_id, payload.flow_id, "pending", json.dumps(payload.input), json.dumps({}), json.dumps([]), created))
    background_tasks.add_task(execute_flow, payload.flow_id, run_id, payload.input)
    return {"run_id": run_id, "status": "started", "created_at": created}

# Very small runner supporting two step types: 'log' and 'http'
def execute_flow(flow_id: str, run_id: str, input_payload: dict):
    started = now_iso()
    db_execute("UPDATE flow_runs SET status = ?, started_at = ? WHERE id = ?", ("running", started, run_id))
    logs = []
    output = {}
    try:
        flow = get_flow(flow_id)
        steps = flow.get("data", {}).get("steps", [])
        context = {"input": input_payload}
        step_results = []
        for idx, step in enumerate(steps):
            stype = step.get("type", "log")
            step_label = step.get("name") or f"step_{idx}"
            if stype == "log":
                msg = step.get("message", "log")
                # support templating simple keys from input using {input.key}
                try:
                    rendered = msg.format(input=input_payload)
                except Exception:
                    rendered = msg
                logs.append({"step": step_label, "type": "log", "message": rendered})
                step_results.append({"step": step_label, "result": rendered})
            elif stype == "http":
                method = (step.get("method") or "GET").upper()
                url = step.get("url")
                headers = step.get("headers") or {}
                body = step.get("body")
                # basic variable replacement for input placeholders {input.key}
                if isinstance(url, str):
                    try:
                        url = url.format(input=input_payload)
                    except Exception:
                        pass
                if isinstance(body, (dict, str)):
                    try:
                        body_str = json.dumps(body) if isinstance(body, dict) else str(body)
                        body_str = body_str.format(input=input_payload)
                        body = json.loads(body_str) if isinstance(body, dict) else body_str
                    except Exception:
                        pass
                logs.append({"step": step_label, "type": "http", "request": {"method": method, "url": url}})
                try:
                    if method == "GET":
                        resp = requests.get(url, headers=headers, timeout=10)
                        res_text = resp.text
                        step_results.append({"step": step_label, "status_code": resp.status_code, "body": resp.text})
                        logs.append({"step": step_label, "type": "http_response", "status": resp.status_code})
                    else:
                        resp = requests.request(method, url, headers=headers, json=body, timeout=10)
                        step_results.append({"step": step_label, "status_code": resp.status_code, "body": resp.text})
                        logs.append({"step": step_label, "type": "http_response", "status": resp.status_code})
                except Exception as e:
                    logs.append({"step": step_label, "type": "http_error", "error": str(e)})
                    raise
            else:
                logs.append({"step": step_label, "type": "unknown", "raw": step})
                step_results.append({"step": step_label, "result": None})
        output = {"steps": step_results, "final": "success"}
        db_execute("UPDATE flow_runs SET status = ?, output = ?, logs = ?, finished_at = ? WHERE id = ?",
                   ("completed", json.dumps(output), json.dumps(logs), now_iso(), run_id))
    except Exception as e:
        logs.append({"error": str(e)})
        db_execute("UPDATE flow_runs SET status = ?, output = ?, logs = ?, finished_at = ? WHERE id = ?",
                   ("failed", json.dumps({"error": str(e)}), json.dumps(logs), now_iso(), run_id))
