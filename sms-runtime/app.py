import os
import json
import time
import re
import traceback
from typing import Dict, Any, Optional, Union
import redis
from fastapi import FastAPI, HTTPException, Request, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import tempfile
import sys
from io import StringIO
from pathlib import Path

# ----------------------------------------------------------------------
# Config & Redis connection (Render env or local fallback)
# ----------------------------------------------------------------------
# Use provided default if REDIS_URL not set in env
redis_url = os.getenv('REDIS_URL', 'redis://red-d44f17jipnbc73dqs2k0:6379')

# Global redis client (may be None if connection fails)
r = None

def _connect_redis(url: str, attempts: int = 2, timeout: float = 2.0):
    global r
    last_err = None
    for i in range(attempts):
        try:
            # decode_responses True so we get string responses
            r_candidate = redis.from_url(url, decode_responses=True, socket_connect_timeout=timeout)
            r_candidate.ping()
            r = r_candidate
            print(f"Connected to Redis at {url}")
            return
        except Exception as e:
            last_err = e
            print(f"Redis connect attempt {i+1}/{attempts} failed: {e}")
            time.sleep(0.2)
    print(f"Redis connection failed after {attempts} attempts: {last_err}")
    r = None

_connect_redis(redis_url)

# Local fallback storage for functions when Redis is not available.
LOCAL_FUNC_DIR = Path(os.getenv("LOCAL_FUNC_DIR", "./local_functions"))
LOCAL_FUNC_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# FastAPI app & CORS
# ----------------------------------------------------------------------
app = FastAPI(title="SMS Runtime Backend")

# CORS for frontend (allows all origins; tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Existing Request Models (Preserved - No Changes)
# ----------------------------------------------------------------------
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
        raise HTTPException(status_code=503, detail="Redis not available. Set REDIS_URL or start Redis.")
    # ping to ensure connection alive
    try:
        r.ping()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Redis not reachable: {e}")

# ----------------------------------------------------------------------
# ONE-TIME INITIALIZATION (Runs on startup if missing)
# ----------------------------------------------------------------------
def _initialize_db():
    """Check and initialize counters/sets if Redis exists."""
    try:
        if not r:
            print("Redis not initialized; skipping DB initialization")
            return
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

# Call initialization now
_initialize_db()

@app.on_event("startup")
def _startup():
    # Reserved for any startup tasks. DB initialization already attempted above.
    pass

# ----------------------------------------------------------------------
# Existing Add (Preserved - No Changes to logic)
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
        r.expire(key, 30 * 86400) # 30 days
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
            if filt:
                msgs = [m for m in msgs if _matches(m, filt)]
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
                if not filt or any(_matches(m, filt) for m in msgs):
                    out[ck] = {"messages": msgs, "meta": meta}
        return out
    raise HTTPException(status_code=400, detail="Invalid table")

# ----------------------------------------------------------------------
# UPDATE (Preserved)
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
# DELETE (Preserved)
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
# ---------------------------
# Table creation endpoint
# ---------------------------
class CreateTableRequest(BaseModel):
    name: str

@app.post("/createtable")
def create_table(req: CreateTableRequest):
    """
    Create a new logical table in Redis.
    Creates a metadata hash at "table_meta:<name>" and an id counter "next_<name>_id".
    Table names are restricted to letters, numbers and underscore to avoid injection.
    """
    name = req.name.strip()
    # validate simple, safe name
    if not name or not re.match(r'^[A-Za-z0-9_]+$', name):
        raise HTTPException(status_code=400, detail="Invalid table name. Use letters, numbers and underscores only.")

    # reserve meta and counter keys
    meta_key = f"table_meta:{name}"
    counter_key = f"next_{name}_id"

    # ensure Redis is available
    _require_redis()

    try:
        # If metadata already exists, treat as existing table
        if r.exists(meta_key) or r.exists(counter_key):
            return {"status": "exists", "table": name}

        # Create metadata and counter. We do not have to create an empty set for members;
        # presence of meta + counter is sufficient to consider the table created.
        now = int(time.time())
        r.hset(meta_key, mapping={
            "name": name,
            "created_at": now,
            "created_by": "api",
        })
        r.set(counter_key, 0)

        return {"status": "created", "table": name, "meta_key": meta_key, "counter_key": counter_key}
    except Exception as e:
        # keep error message concise and useful for debugging
        print(f"Error creating table {name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create table: {str(e)}")

# ----------------------------------------------------------------------
# FUNCTION MANAGEMENT (Updated to handle Redis unavailability gracefully)
# ----------------------------------------------------------------------
def _sanitize_code(code: str) -> str:
    """Basic code sanitization - block dangerous keywords."""
    # Use word boundaries for tokens to reduce false positives
    dangerous_tokens = [
        r'\beval\b', r'\bexec\b', r'\bcompile\b', r'__import__', r'\bopen\b',
        r'\bos\.system\b', r'\bsubprocess\b', r'\bimport \*\b', r'\bsocket\b'
    ]
    lowered = code.lower()
    for token in dangerous_tokens:
        if re.search(token, lowered):
            raise ValueError(f"Unsafe code detected: {token}")
    return code

def _run_code(code: str, inputs: Dict[str, Any], language: str = "python") -> Dict[str, Any]:
    """Run code safely. Python only."""
    if language != "python":
        raise ValueError("Only Python supported")

    # Minimal builtins whitelist: allow basic builtins, but remove dangerous ones
    safe_builtins = {}
    for k, v in __builtins__.items():
        # keep only commonly safe names (int, str, len, dict, list, etc.)
        if k in ("abs", "all", "any", "bool", "dict", "float", "int", "len", "list", "max", "min", "range", "str", "sum", "print"):
            safe_builtins[k] = v

    globals_dict = {"__builtins__": safe_builtins}
    locals_dict = inputs.copy() if isinstance(inputs, dict) else {}
    try:
        exec(code, globals_dict, locals_dict)
        # Return only JSON-serializable items: attempt to convert non-serializable to str
        def _serialize(o):
            try:
                json.dumps(o)
                return o
            except Exception:
                return str(o)
        serialized = {k: _serialize(v) for k, v in locals_dict.items()}
        return {"locals": serialized}
    except Exception as e:
        tb = traceback.format_exc()
        return {"error": str(e), "traceback": tb}

def _save_local_function(fid: str, code: str, meta: Dict[str, Any]):
    code_path = LOCAL_FUNC_DIR / f"{fid}.py"
    meta_path = LOCAL_FUNC_DIR / f"{fid}.meta.json"
    code_path.write_text(code, encoding="utf-8")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

def _load_local_function(fid: str):
    code_path = LOCAL_FUNC_DIR / f"{fid}.py"
    meta_path = LOCAL_FUNC_DIR / f"{fid}.meta.json"
    if code_path.exists():
        code = code_path.read_text(encoding="utf-8")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        return code, meta
    return None, None

@app.post("/createfunction")
async def create_function(
    file: UploadFile = File(...),
    language: str = Form("python"),
    name: str = Form(...)
):
    # NOTE: Do not require Redis here. Allow local fallback so user can upload even when Redis down.
    if not file.filename.endswith('.py'):
        raise HTTPException(400, detail="Only .py files supported")

    content = await file.read()
    code = content.decode('utf-8')
    code = _sanitize_code(code)

    # Analyze for libs (check imports, warn if unknown)
    libs = []
    for line in code.split('\n'):
        l = line.strip()
        if l.startswith('import ') or l.startswith('from '):
            lib_match = re.match(r'(?:from|import)\s+([A-Za-z0-9_]+)', l)
            if lib_match:
                lib = lib_match.group(1)
                if lib not in libs:
                    libs.append(lib)
                    if lib not in ['redis', 'json', 'time', 'typing', 'fastapi', 'pydantic']:
                        print(f"Warning: Unknown lib '{lib}' - pre-install in requirements.txt")

    # Generate ID
    ts = int(time.time())
    fid = f"func{ts}"

    meta = {
        "name": name,
        "language": language,
        "libs": libs,
        "ts": ts
    }

    # Try to store in Redis; if Redis not available, store locally
    if r:
        try:
            r.set(f"function:{fid}:code", code)
            r.hset(f"function:{fid}:meta", mapping={
                "name": name,
                "language": language,
                "libs": json.dumps(libs),
                "ts": ts
            })
            return {"id": fid, "status": "created", "libs": libs}
        except Exception as e:
            # Log and fall back to local
            print(f"Redis write failed while creating function {fid}: {e}")
            _save_local_function(fid, code, meta)
            return {"id": fid, "status": "created_local", "note": "Redis write failed; saved locally", "libs": libs}
    else:
        _save_local_function(fid, code, meta)
        return {"id": fid, "status": "created_local", "note": "Redis not available; saved locally", "libs": libs}

@app.post("/runfunction")
async def run_function(request: Request):
    # Allow running functions even if Redis is down, if saved locally
    data = await request.json()
    fid = data.get("id")
    inputs = data.get("input", {})

    if not fid:
        raise HTTPException(400, detail="Missing 'id'")

    code = None
    meta = {}
    if r:
        try:
            code = r.get(f"function:{fid}:code")
            if code:
                meta = dict(r.hgetall(f"function:{fid}:meta") or {})
        except Exception as e:
            print(f"Redis read error for {fid}: {e}")
            code = None

    if not code:
        # Attempt local load
        code_local, meta_local = _load_local_function(fid)
        if code_local:
            code = code_local
            meta = meta_local or meta
    if not code:
        raise HTTPException(status_code=404, detail="Function not found")

    language = meta.get("language", "python") if meta else "python"
    result = _run_code(code, inputs, language)
    return {"result": result, "status": "ran"}

@app.post("/updatefunction")
async def update_function(
    id: str = Form(...),
    file: UploadFile = File(...),
    name: str = Form(...)
):
    if not file.filename.endswith('.py'):
        raise HTTPException(400, detail="Only .py files supported")

    content = await file.read()
    code = content.decode('utf-8')
    code = _sanitize_code(code)

    # Analyze libs (same as create)
    libs = []
    for line in code.split('\n'):
        l = line.strip()
        if l.startswith('import ') or l.startswith('from '):
            lib_match = re.match(r'(?:from|import)\s+([A-Za-z0-9_]+)', l)
            if lib_match:
                lib = lib_match.group(1)
                if lib not in libs:
                    libs.append(lib)
                    if lib not in ['redis', 'json', 'time', 'typing', 'fastapi', 'pydantic']:
                        print(f"Warning: New lib '{lib}' - add to requirements.txt and redeploy")

    meta = {
        "name": name,
        "libs": libs,
        "ts": int(time.time())
    }

    if r:
        try:
            r.set(f"function:{id}:code", code)
            r.hset(f"function:{id}:meta", mapping={
                "name": name,
                "libs": json.dumps(libs),
                "ts": int(time.time())
            })
            return {"id": id, "status": "updated", "libs": libs}
        except Exception as e:
            print(f"Redis update failed for {id}: {e}")
            _save_local_function(id, code, meta)
            return {"id": id, "status": "updated_local", "note": "Redis update failed; saved locally", "libs": libs}
    else:
        _save_local_function(id, code, meta)
        return {"id": id, "status": "updated_local", "note": "Redis not available; saved locally", "libs": libs}

@app.delete("/deletefunction")
def delete_function(id: str):
    # Delete from Redis if possible; otherwise delete local files
    deleted = False
    if r:
        try:
            r_deleted = r.delete(f"function:{id}:code")
            r_meta_deleted = r.delete(f"function:{id}:meta")
            deleted = bool(r_deleted or r_meta_deleted)
        except Exception as e:
            print(f"Redis delete error for {id}: {e}")
            deleted = False

    # Remove local fallback files
    code_path = LOCAL_FUNC_DIR / f"{id}.py"
    meta_path = LOCAL_FUNC_DIR / f"{id}.meta.json"
    local_deleted = False
    try:
        if code_path.exists():
            code_path.unlink()
            local_deleted = True
        if meta_path.exists():
            meta_path.unlink()
            local_deleted = True
    except Exception as e:
        print(f"Local delete error for {id}: {e}")

    if deleted or local_deleted:
        return {"id": id, "status": "deleted"}
    raise HTTPException(status_code=404, detail="Function not found")

# ----------------------------------------------------------------------
# Existing routes preserved (no changes)
# Note: /add, /fetch, /update, /delete endpoints are above and unchanged
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
