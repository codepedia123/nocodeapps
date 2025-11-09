import os
import json
import time
from typing import Dict, Any, Optional, Union
import redis
from fastapi import FastAPI, HTTPException, Request, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import tempfile
import re
import sys
from io import StringIO
from contextlib import redirect_stdout

# Redis connection (Render env or local fallback)
redis_url = os.getenv('REDIS_URL', 'redis://red-d44f17jipnbc73dqs2k0:6379')
try:
    r = redis.from_url(redis_url, decode_responses=True)
    r.ping()  # Test connection
except redis.ConnectionError as e:
    print(f"Redis connection failed: {e}. Check REDIS_URL or start local Docker.")
    r = None  # Set to None; routes handle gracefully

app = FastAPI(title="SMS Runtime Backend")

# CORS for frontend (allows all origins; tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Existing Request Models (Preserved - No Changes)
class AddRequest(BaseModel):
    table: str
    data: Dict[str, Any]

class UpdateRequest(BaseModel):
    table: str
    id: str
    updates: Dict[str, Any]

# New Request Models for Function Management
class FetchFunctionRequest(BaseModel):
    id: str
    input: Dict[str, Any]

class DeleteFunctionRequest(BaseModel):
    id: str

# ----------------------------------------------------------------------
# Helper: Redis Check
# ----------------------------------------------------------------------
def _require_redis():
    if not r:
        raise HTTPException(status_code=503, detail="Redis not available")

# ----------------------------------------------------------------------
# One-Time Initialization (Runs on startup if missing)
# ----------------------------------------------------------------------
def _initialize_db():
    """Check and initialize counters/sets if not exist."""
    try:
        if not r.exists("next_user_id"):
            r.set("next_user_id", "0")
            print("Initialized next_user_id")
        if not r.exists("next_agent_id"):
            r.set("next_agent_id", "0")
            print("Initialized next_agent_id")
        # Sets are lazy - no need to create empty
        print("DB initialized successfully")
    except Exception as e:
        print(f"DB init error: {e}")

# Run on startup
@app.on_event("startup")
def startup_event():
    _initialize_db()

# ----------------------------------------------------------------------
# Existing Add (Preserved - No Changes)
# ----------------------------------------------------------------------
def add_record(table: str, data: Dict[str, Any]) -> str:
    """Add new record with auto-ID and timestamps; accepts any fields."""
    if not r:
        raise ValueError("Redis not connected")
    data["created_at"] = int(time.time())
    data["updated_at"] = data["created_at"]

    if table == "users":
        next_id = int(r.get("next_user_id") or 0) + 1
        r.set("next_user_id", next_id)
        user_id = str(next_id)
        key = f"user:{user_id}"
        r.hset(key, mapping=data)
        r.sadd("users", user_id)
        return user_id

    elif table == "agents":
        next_id = int(r.get("next_agent_id") or 0) + 1
        r.set("next_agent_id", next_id)
        agent_id = f"agent{next_id}"
        if "tools" in data:
            data["tools"] = json.dumps(data["tools"])
        r.hset(agent_id, mapping=data)
        r.sadd("agents", agent_id)
        return agent_id

    elif table == "conversations":
        agent_id = data.get("agent_id")
        phone = data.get("phone")
        if not agent_id or not phone:
            raise ValueError("agent_id and phone required for conversations")
        key = f"convo:{agent_id}:{phone}"
        meta_key = f"convo_meta:{agent_id}:{phone}"
        messages = data.pop("messages", [])
        r.delete(key)
        for msg in messages:
            r.rpush(key, json.dumps(msg))
        r.expire(key, 86400 * 30)
        r.hset(meta_key, mapping=data)
        r.sadd("conversations", f"{agent_id}:{phone}")
        return f"{agent_id}:{phone}"

    raise ValueError(f"Invalid table: {table}")

@app.post("/add")
def add_endpoint(req: AddRequest):
    _require_redis()
    data = req.data.copy()
    ts = int(time.time())
    data["created_at"] = ts
    data["updated_at"] = ts
    if req.table == "users":
        next_id = int(r.get("next_user_id") or 0) + 1
        r.set("next_user_id", next_id)
        user_id = str(next_id)
        key = f"user:{user_id}"
        r.hset(key, mapping=data)
        r.sadd("users", user_id)
        return {"id": user_id, "status": "success"}
    if req.table == "agents":
        next_id = int(r.get("next_agent_id") or 0) + 1
        r.set("next_agent_id", next_id)
        agent_id = f"agent{next_id}"
        if "tools" in data:
            data["tools"] = json.dumps(data["tools"])
        r.hset(agent_id, mapping=data)
        r.sadd("agents", agent_id)
        return {"id": agent_id, "status": "success"}
    if req.table == "conversations":
        agent_id = data.get("agent_id")
        phone = data.get("phone")
        if not agent_id or not phone:
            raise HTTPException(status_code=400, detail="agent_id and phone required")
        key = f"convo:{agent_id}:{phone}"
        meta_key = f"convo_meta:{agent_id}:{phone}"
        messages = data.pop("messages", [])
        r.delete(key)
        for msg in messages:
            r.rpush(key, json.dumps(msg))
        r.expire(key, 30 * 86400)  # 30 days
        r.hset(meta_key, mapping=data)
        r.sadd("conversations", f"{agent_id}:{phone}")
        return {"id": f"{agent_id}:{phone}", "status": "success"}
    raise HTTPException(status_code=400, detail="Invalid table")

# ----------------------------------------------------------------------
# Existing Fetch (Preserved - No Changes)
# ----------------------------------------------------------------------
def _parse_filters(filter_str: Optional[str]) -> Dict[str, str]:
    filters = {}
    if filter_str:
        for part in filter_str.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                filters[k.strip()] = v.strip()
    return filters
def _matches(record: Dict[str, Any], filters: Dict[str, str]) -> bool:
    if not filters:
        return True
    for k, v in filters.items():
        if str(record.get(k, "")).lower() != v.lower():
            return False
    return True
@app.get("/fetch")
def fetch_endpoint(table: str, id: Optional[str] = None, filters: Optional[str] = None):
    _require_redis()
    filt = _parse_filters(filters)
    # USERS
    if table == "users":
        if id:
            rec = dict(r.hgetall(f"user:{id}"))
            if not rec:
                return {}
            rec.setdefault("name", "")
            rec.setdefault("phone", "")
            rec["last_active"] = int(rec.get("last_active", 0))
            rec["created_at"] = int(rec.get("created_at", 0))
            return rec if _matches(rec, filt) else {}
        out = {}
        for uid in r.smembers("users"):
            rec = dict(r.hgetall(f"user:{uid}"))
            if rec and _matches(rec, filt):
                rec.setdefault("name", "")
                rec.setdefault("phone", "")
                rec["last_active"] = int(rec.get("last_active", 0))
                rec["created_at"] = int(rec.get("created_at", 0))
                out[uid] = rec
        return out
    # AGENTS
    if table == "agents":
        if id:
            rec = dict(r.hgetall(id))
            if not rec:
                return {}
            rec.setdefault("prompt", "")
            rec.setdefault("user_id", "")
            rec["tools"] = json.loads(rec["tools"]) if "tools" in rec else []
            rec["created_at"] = int(rec.get("created_at", 0))
            rec["updated_at"] = int(rec.get("updated_at", 0))
            return rec if _matches(rec, filt) else {}
        out = {}
        for aid in r.smembers("agents"):
            rec = dict(r.hgetall(aid))
            if rec and _matches(rec, filt):
                rec.setdefault("prompt", "")
                rec.setdefault("user_id", "")
                rec["tools"] = json.loads(rec["tools"]) if "tools" in rec else []
                rec["created_at"] = int(rec.get("created_at", 0))
                rec["updated_at"] = int(rec.get("updated_at", 0))
                out[aid] = rec
        return out
    # CONVERSATIONS
    if table == "conversations":
        if id:
            try:
                agent_id, phone = id.split(":", 1)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid conversation id")
            key = f"convo:{agent_id}:{phone}"
            msgs = [json.loads(m) for m in r.lrange(key, 0, -1)]
            meta_key = f"convo_meta:{agent_id}:{phone}"
            meta = dict(r.hgetall(meta_key))
            meta.setdefault("agent_id", agent_id)
            meta.setdefault("phone", phone)
            meta["created_at"] = int(meta.get("created_at", 0))
            meta["updated_at"] = int(meta.get("updated_at", 0))
            if filters:
                msgs = [m for m in msgs if _matches(m, filters)]
                if not msgs:
                    return {"messages": [], "meta": meta}
            return {"messages": msgs, "meta": meta}
        out = {}
        for ck in r.smembers("conversations"):
            agent_id, phone = ck.split(":", 1)
            key = f"convo:{agent_id}:{phone}"
            msgs = [json.loads(m) for m in r.lrange(key, 0, -1)]
            meta_key = f"convo_meta:{agent_id}:{phone}"
            meta = dict(r.hgetall(meta_key))
            if msgs or meta:
                meta.setdefault("agent_id", agent_id)
                meta.setdefault("phone", phone)
                meta["created_at"] = int(meta.get("created_at", 0))
                meta["updated_at"] = int(meta.get("updated_at", 0))
                if not filters or any(_matches(m, filters) for m in msgs):
                    out[ck] = {"messages": msgs, "meta": meta}
        return out
    raise HTTPException(status_code=400, detail="Invalid table")

# ----------------------------------------------------------------------
# UPDATE
# ----------------------------------------------------------------------
@app.post("/update")
def update_endpoint(req: UpdateRequest):
    _require_redis()
    ts = int(time.time())
    updates = req.updates.copy()
    updates["updated_at"] = ts

    if req.table == "users":
        key = f"user:{req.id}"
        if not r.exists(key):
            raise HTTPException(status_code=404, detail="User not found")
        r.hset(key, mapping=updates)
        return {"status": "success"}

    if req.table == "agents":
        if not r.exists(req.id):
            raise HTTPException(status_code=404, detail="Agent not found")
        if "tools" in updates:
            updates["tools"] = json.dumps(updates["tools"])
        r.hset(req.id, mapping=updates)
        return {"status": "success"}

    if req.table == "conversations":
        try:
            agent_id, phone = req.id.split(":", 1)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid conversation id")
        meta_key = f"convo_meta:{agent_id}:{phone}"
        msg_key = f"convo:{agent_id}:{phone}"
        if not r.exists(meta_key) and not r.exists(msg_key):
            raise HTTPException(status_code=404, detail="Conversation not found")
        if "append" in updates:
            for m in updates.pop("append"):
                r.rpush(msg_key, json.dumps(m))
        if updates:
            r.hset(meta_key, mapping=updates)
        return {"status": "success"}

    raise HTTPException(status_code=400, detail="Invalid table")

# ----------------------------------------------------------------------
# DELETE
# ----------------------------------------------------------------------
@app.delete("/delete")
def delete_endpoint(table: str, id: str):
    _require_redis()

    if table == "users":
        key = f"user:{id}"
        if r.delete(key):
            r.srem("users", id)
            return {"status": "deleted"}
        raise HTTPException(status_code=404, detail="User not found")

    if table == "agents":
        if r.delete(id):
            r.srem("agents", id)
            return {"status": "deleted"}
        raise HTTPException(status_code=404, detail="Agent not found")

    if table == "conversations":
        try:
            agent_id, phone = id.split(":", 1)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid conversation id")
        msg_key = f"convo:{agent_id}:{phone}"
        meta_key = f"convo_meta:{agent_id}:{phone}"
        deleted = r.delete(msg_key, meta_key)
        if deleted:
            r.srem("conversations", id)
            return {"status": "deleted"}
        raise HTTPException(status_code=404, detail="Conversation not found")

    raise HTTPException(status_code=400, detail="Invalid table")

# ----------------------------------------------------------------------
# FUNCTION MANAGEMENT
# ----------------------------------------------------------------------
def _sanitize_code(code: str) -> str:
    """Basic code sanitization - block dangerous keywords."""
    dangerous = ['exec', 'eval', 'compile', 'open', '__import__', 'import *', 'os.system', 'subprocess', 'sys.exit']
    for d in dangerous:
        if d in code.lower():
            raise ValueError(f"Unsafe code detected: {d}")
    return code

def _run_python_code(code: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Run code safely. Python only."""
    # Isolated execution
    globals_dict = {"__builtins__": {k: v for k, v in __builtins__.items() if k not in ['eval', 'exec', 'compile']}}
    locals_dict = inputs.copy()
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(code, globals_dict, locals_dict)
        output = locals_dict
        output["stdout"] = sys.stdout.getvalue()
        return output
    except Exception as e:
        return {"error": str(e), "stdout": sys.stdout.getvalue()}
    finally:
        sys.stdout = old_stdout

@app.post("/createfunction")
async def create_function(
    file: UploadFile,
    language: str = Form("python"),
    name: str = Form(...)
):
    _require_redis()
    if not file.filename.endswith('.py'):
        raise HTTPException(400, detail="Only .py files supported")
    
    content = await file.read()
    code = content.decode('utf-8')
    code = _sanitize_code(code)
    
    # Analyze for libs (check imports, warn if unknown - no dynamic pip)
    libs = []
    for line in code.split('\n'):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            # Extract lib name (e.g., "import requests" → "requests")
            lib_match = re.match(r'(?:from|import)\s+(\w+)', line)
            if lib_match:
                lib = lib_match.group(1)
                if lib not in libs and lib not in ['redis', 'json', 'time', 'typing', 'fastapi', 'pydantic']:
                    libs.append(lib)
                    print(f"Warning: Unknown lib '{lib}' - pre-install in requirements.txt and redeploy")
    
    # Generate ID
    ts = int(time.time())
    fid = f"func{ts}"
    
    # Store in Redis
    r.set(f"function:{fid}:code", code)
    r.hset(f"function:{fid}:meta", mapping={
        "name": name,
        "language": language,
        "libs": json.dumps(libs),
        "ts": ts
    })
    
    return {"id": fid, "status": "created", "libs": libs}

@app.post("/fetchfunction")
async def fetch_function(request: Request):
    _require_redis()
    data = await request.json()
    fid = data.get("id")
    inputs = data.get("input", {})
    
    if not fid:
        raise HTTPException(400, detail="Missing 'id'")
    
    code = r.get(f"function:{fid}:code")
    if not code:
        raise HTTPException(404, detail="Function not found")
    
    meta = dict(r.hgetall(f"function:{fid}:meta"))
    language = meta.get("language", "python")
    
    result = _run_python_code(code, inputs)
    return {"result": result, "status": "ran"}

@app.post("/updatefunction")
async def update_function(
    id: str = Form(...),
    file: UploadFile,
    name: str = Form(...)
):
    _require_redis()
    if not file.filename.endswith('.py'):
        raise HTTPException(400, detail="Only .py files supported")
    
    content = await file.read()
    code = content.decode('utf-8')
    code = _sanitize_code(code)
    
    # Analyze libs (same as create)
    libs = []
    for line in code.split('\n'):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            lib_match = re.match(r'(?:from|import)\s+(\w+)', line)
            if lib_match:
                lib = lib_match.group(1)
                if lib not in libs and lib not in ['redis', 'json', 'time', 'typing', 'fastapi', 'pydant']:
                    libs.append(lib)
                    print(f"Warning: New lib '{lib}' - add to requirements.txt and redeploy")
    
    # Update Redis
    r.set(f"function:{id}:code", code)
    r.hset(f"function:{id}:meta", mapping={
        "name": name,
        "libs": json.dumps(libs),
        "ts": int(time.time())
    })
    
    return {"id": id, "status": "updated", "libs": libs}

@app.delete("/deletefunction")
def delete_function(id: str):
    _require_redis()
    r.delete(f"function:{id}:code")
    r.delete(f"function:{id}:meta")
    return {"id": id, "status": "deleted"}

# Existing routes preserved (no changes)
# ----------------------------------------------------------------------
# Fetch
# ----------------------------------------------------------------------
def _parse_filters(filter_str: Optional[str]) -> Dict[str, str]:
    filters = {}
    if filter_str:
        for part in filter_str.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                filters[k.strip()] = v.strip()
    return filters
def _matches(record: Dict[str, Any], filters: Dict[str, str]) -> bool:
    if not filters:
        return True
    for k, v in filters.items():
        if str(record.get(k, "")).lower() != v.lower():
            return False
    return True
@app.get("/fetch")
def fetch_endpoint(table: str, id: Optional[str] = None, filters: Optional[str] = None):
    _require_redis()
    filt = _parse_filters(filters)
    # USERS
    if table == "users":
        if id:
            rec = dict(r.hgetall(f"user:{id}"))
            if not rec:
                return {}
            rec.setdefault("name", "")
            rec.setdefault("phone", "")
            rec["last_active"] = int(rec.get("last_active", 0))
            rec["created_at"] = int(rec.get("created_at", 0))
            return rec if _matches(rec, filt) else {}
        out = {}
        for uid in r.smembers("users"):
            rec = dict(r.hgetall(f"user:{uid}"))
            if rec and _matches(rec, filt):
                rec.setdefault("name", "")
                rec.setdefault("phone", "")
                rec["last_active"] = int(rec.get("last_active", 0))
                rec["created_at"] = int(rec.get("created_at", 0))
                out[uid] = rec
        return out
    # AGENTS
    if table == "agents":
        if id:
            rec = dict(r.hgetall(id))
            if not rec:
                return {}
            rec.setdefault("prompt", "")
            rec.setdefault("user_id", "")
            rec["tools"] = json.loads(rec["tools"]) if "tools" in rec else []
            rec["created_at"] = int(rec.get("created_at", 0))
            rec["updated_at"] = int(rec.get("updated_at", 0))
            return rec if _matches(rec, filt) else {}
        out = {}
        for aid in r.smembers("agents"):
            rec = dict(r.hgetall(aid))
            if rec and _matches(rec, filt):
                rec.setdefault("prompt", "")
                rec.setdefault("user_id", "")
                rec["tools"] = json.loads(rec["tools"]) if "tools" in rec else []
                rec["created_at"] = int(rec.get("created_at", 0))
                rec["updated_at"] = int(rec.get("updated_at", 0))
                out[aid] = rec
        return out
    # CONVERSATIONS
    if table == "conversations":
        if id:
            try:
                agent_id, phone = id.split(":", 1)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid conversation id")
            key = f"convo:{agent_id}:{phone}"
            msgs = [json.loads(m) for m in r.lrange(key, 0, -1)]
            meta_key = f"convo_meta:{agent_id}:{phone}"
            meta = dict(r.hgetall(meta_key))
            meta.setdefault("agent_id", agent_id)
            meta.setdefault("phone", phone)
            meta["created_at"] = int(meta.get("created_at", 0))
            meta["updated_at"] = int(meta.get("updated_at", 0))
            if filters:
                msgs = [m for m in msgs if _matches(m, filters)]
                if not msgs:
                    return {"messages": [], "meta": meta}
            return {"messages": msgs, "meta": meta}
        out = {}
        for ck in r.smembers("conversations"):
            agent_id, phone = ck.split(":", 1)
            key = f"convo:{agent_id}:{phone}"
            msgs = [json.loads(m) for m in r.lrange(key, 0, -1)]
            meta_key = f"convo_meta:{agent_id}:{phone}"
            meta = dict(r.hgetall(meta_key))
            if msgs or meta:
                meta.setdefault("agent_id", agent_id)
                meta.setdefault("phone", phone)
                meta["created_at"] = int(meta.get("created_at", 0))
                meta["updated_at"] = int(meta.get("updated_at", 0))
                if not filters or any(_matches(m, filters) for m in msgs):
                    out[ck] = {"messages": msgs, "meta": meta}
        return out
    raise HTTPException(status_code=400, detail="Invalid table")
# ----------------------------------------------------------------------
# UPDATE
# ----------------------------------------------------------------------
@app.post("/update")
def update_endpoint(req: UpdateRequest):
    _require_redis()
    ts = int(time.time())
    updates = req.updates.copy()
    updates["updated_at"] = ts
    if req.table == "users":
        key = f"user:{req.id}"
        if not r.exists(key):
            raise HTTPException(status_code=404, detail="User not found")
        r.hset(key, mapping=updates)
        return {"status": "success"}
    if req.table == "agents":
        if not r.exists(req.id):
            raise HTTPException(status_code=404, detail="Agent not found")
        if "tools" in updates:
            updates["tools"] = json.dumps(updates["tools"])
        r.hset(req.id, mapping=updates)
        return {"status": "success"}
    if req.table == "conversations":
        try:
            agent_id, phone = req.id.split(":", 1)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid conversation id")
        meta_key = f"convo_meta:{agent_id}:{phone}"
        msg_key = f"convo:{agent_id}:{phone}"
        if not r.exists(meta_key) and not r.exists(msg_key):
            raise HTTPException(status_code=404, detail="Conversation not found")
        if "append" in updates:
            for m in updates.pop("append"):
                r.rpush(msg_key, json.dumps(m))
        if updates:
            r.hset(meta_key, mapping=updates)
        return {"status": "success"}
    raise HTTPException(status_code=400, detail="Invalid table")
# ----------------------------------------------------------------------
# DELETE
# ----------------------------------------------------------------------
@app.delete("/delete")
def delete_endpoint(table: str, id: str):
    _require_redis()
    if table == "users":
        key = f"user:{id}"
        if r.delete(key):
            r.srem("users", id)
            return {"status": "deleted"}
        raise HTTPException(status_code=404, detail="User not found")
    if table == "agents":
        if r.delete(id):
            r.srem("agents", id)
            return {"status": "deleted"}
        raise HTTPException(status_code=404, detail="Agent not found")
    if table == "conversations":
        try:
            agent_id, phone = id.split(":", 1)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid conversation id")
        msg_key = f"convo:{agent_id}:{phone}"
        meta_key = f"convo_meta:{agent_id}:{phone}"
        deleted = r.delete(msg_key, meta_key)
        if deleted:
            r.srem("conversations", id)
            return {"status": "deleted"}
        raise HTTPException(status_code=404, detail="Conversation not found")
    raise HTTPException(status_code=400, detail="Invalid table")
# ----------------------------------------------------------------------
# Run (Render uses: uvicorn sms-runtime.app:app)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
