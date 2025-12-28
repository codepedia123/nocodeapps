# sms-runtime/app.py
import os
import json
import time
import uuid
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List, Tuple

# --- Redis Cluster Supporrt ---
from redis.cluster import RedisCluster
import redis
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import traceback
import re
import builtins
import threading
load_dotenv()
# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
def _init_redis() -> Optional[redis.Redis]:
    try:
        # 1. Prioritize Railway's automatic environment variable
        # 2. Fallback to hardcoded AWS URL if not in Railway
        redis_url = os.getenv("REDIS_URL", "redis://default:AdvvAAIncDExZmMzYTBiNTJhZWU0MzA1YjA1M2IwYWU4NThlZjcyM3AxNTYzMDM@climbing-hyena-56303.upstash.io:6379")

        # Railway Redis is standalone (needs redis.from_url)
        # AWS Serverless is a Cluster (needs RedisCluster.from_url)
        if "cache.amazonaws.com" in redis_url:
            print("🔗 Connecting to AWS Redis Cluster...")
            client = RedisCluster.from_url(
                redis_url,
                decode_responses=True,
                ssl_cert_reqs=None,
                socket_connect_timeout=3,
                socket_timeout=3,
                read_from_replicas=True
            )
        else:
            print("🔗 Connecting to Standalone Redis (Railway)...")
            client = redis.from_url(
                redis_url, 
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3
            )

        client.ping()
        print("✅ Redis connection successful")
        return client

    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return None


r = _init_redis()
MAX_FETCH_KEYS = int(os.getenv("MAX_FETCH_KEYS", "5000"))
safe_builtins = {
    "abs": abs, "all": all, "any": any, "bool": bool,
    "dict": dict, "float": float, "int": int, "len": len,
    "list": list, "max": max, "min": min, "range": range,
    "str": str, "sum": sum, "print": print,
    "Exception": Exception
}

app = FastAPI(title="SMS Runtime Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Request models
# ----------------------------------------------------------------------
class AddRequest(BaseModel):
    table: str
    data: Dict[str, Any]
    
class TableRequest(BaseModel):
    name: str

class UpdateRequest(BaseModel):
    table: str
    id: str
    updates: Dict[str, Any]

class BulkDeleteRequest(BaseModel):
    ids: List[str]


# ----------------------------------------------------------------------
# Key utilities and helpers
# ----------------------------------------------------------------------
def _require_redis():
    if not r:
        raise HTTPException(status_code=503, detail="Redis not available")

def _get_next_user_id() -> int:
    return int(r.get("next_user_id") or 0)

def _get_next_agent_id() -> int:
    return int(r.get("next_agent_id") or 0)

def _user_key(uid: int) -> str:
    return f"user:{uid}"

def _agent_name(aid_num: int) -> str:
    return f"agent{aid_num}"

def _convo_msg_key(agent_name: str, phone: str) -> str:
    return f"convo:{agent_name}:{phone}"

def _convo_meta_key(agent_name: str, phone: str) -> str:
    return f"convo_meta:{agent_name}:{phone}"

# --- NEW: Table helpers for correct Redis key management ---
def _table_meta_key(name: str) -> str:
    return f"table:{name}:meta"

def _table_ids_key(name: str) -> str:
    return f"table:{name}:ids"

def _table_row_key(name: str, rowid: str) -> str:
    return f"table:{name}:row:{rowid}"
def _list_all_tables_with_counts() -> List[Dict[str, Any]]:
    """
    Returns all tables with record counts.
    Covers system tables and dynamic tables.
    """
    tables = []

    # ---- System tables ----
    try:
        users_count = r.scard("users")
        tables.append({
            "name": "users",
            "records": int(users_count or 0)
        })
    except Exception:
        pass

    try:
        agents_count = r.scard("agents")
        tables.append({
            "name": "agents",
            "records": int(agents_count or 0)
        })
    except Exception:
        pass

    try:
        conversations_count = r.scard("conversations")
        tables.append({
            "name": "conversations",
            "records": int(conversations_count or 0)
        })
    except Exception:
        pass

    # ---- Dynamic tables ----
    try:
        dynamic_tables = r.smembers("tables") or []
        for name in sorted(dynamic_tables):
            ids_key = _table_ids_key(name)
            count = r.scard(ids_key) or 0

            # subtract sentinel "_meta"
            if r.sismember(ids_key, "_meta"):
                count -= 1

            tables.append({
                "name": name,
                "records": max(count, 0)
            })
    except Exception:
        pass

    return tables
@app.get("/tables")
def list_tables():
    _require_redis()
    return {
        "tables": _list_all_tables_with_counts()
    }
def _table_exists(name: str) -> bool:
    return r.sismember("tables", name)

def _create_table(name: str):
    if _table_exists(name):
        raise HTTPException(status_code=400, detail="Table already exists")
    r.sadd("tables", name)
    r.hset(_table_meta_key(name), mapping={"created_at": int(time.time())})
    # use a dedicated set for ids and add a sentinel value to ensure set type
    r.sadd(_table_ids_key(name), "_meta")
    r.set(f"nextid:{name}", 0)
    return True

def _delete_table(name: str):
    if not _table_exists(name):
        raise HTTPException(status_code=404, detail="Table not found")
    # Delete all rows for this table
    ids_key = _table_ids_key(name)
    row_ids = r.smembers(ids_key)
    for row_id in row_ids:
        if row_id != "_meta":
            r.delete(_table_row_key(name, row_id))
    r.delete(_table_meta_key(name))
    r.delete(ids_key)
    r.srem("tables", name)
    r.delete(f"nextid:{name}")
    return True

# get numeric suffix ids for keys like user:N or agent:N
def _all_numeric_suffix_ids(prefix: str) -> List[int]:
    ids = []
    pattern = f"{prefix}:*"
    for key in r.scan_iter(match=pattern):
        parts = key.split(":")
        if len(parts) >= 2:
            suffix = parts[1]
            if suffix.isdigit():
                ids.append(int(suffix))
    return sorted(set(ids))

# ----------------------------------------------------------------------
# Compaction helpers (unchanged behavior but included)
# ----------------------------------------------------------------------
def compact_users() -> Dict[str, Any]:
    ids = _all_numeric_suffix_ids("user")
    if not ids:
        r.set("next_user_id", 0)
        r.delete("users")
        return {"status": "ok", "users_before": 0, "users_after": 0}
    mapping: Dict[int, int] = {}
    for new_idx, old_id in enumerate(ids, start=1):
        mapping[old_id] = new_idx
    temp_map: Dict[str, Tuple[int, int]] = {}
    uid_uuid = uuid.uuid4().hex
    for old_id, new_id in mapping.items():
        if old_id == new_id:
            continue
        old_key = _user_key(old_id)
        if not r.exists(old_key):
            continue
        temp_key = f"tmp:rekey:user:{uid_uuid}:{old_id}"
        if r.exists(temp_key):
            r.delete(temp_key)
        r.rename(old_key, temp_key)
        temp_map[temp_key] = (old_id, new_id)
    final_ids: List[int] = []
    for old_id, new_id in mapping.items():
        if old_id == new_id:
            key = _user_key(old_id)
            if r.exists(key):
                final_ids.append(new_id)
    for temp_key, (old_id, new_id) in temp_map.items():
        final_key = _user_key(new_id)
        if r.exists(final_key):
            r.delete(final_key)
        r.rename(temp_key, final_key)
        final_ids.append(new_id)
    final_ids = sorted(set(final_ids))
    r.delete("users")
    for uid in final_ids:
        r.sadd("users", str(uid))
    old_to_new = {old: new for old, new in mapping.items() if old != new}
    for aid in list(r.smembers("agents")):
        try:
            user_id = r.hget(aid, "user_id")
            if not user_id:
                continue
            if user_id.isdigit():
                old_uid = int(user_id)
                if old_uid in old_to_new:
                    r.hset(aid, "user_id", str(old_to_new[old_uid]))
        except Exception:
            continue
    r.set("next_user_id", len(final_ids))
    return {"status": "ok", "users_before": len(ids), "users_after": len(final_ids)}

def _collect_conversations_for_agent(agent_name: str) -> List[str]:
    out = []
    for ck in r.smembers("conversations"):
        if not ck:
            continue
        if ck.startswith(agent_name + ":"):
            out.append(ck)
    return out

def compact_agents() -> Dict[str, Any]:
    ids = _all_numeric_suffix_ids("agent")
    if not ids:
        r.set("next_agent_id", 0)
        r.delete("agents")
        return {"status": "ok", "agents_before": 0, "agents_after": 0}
    mapping: Dict[int, int] = {}
    for new_idx, old_id in enumerate(ids, start=1):
        mapping[old_id] = new_idx
    agent_uuid = uuid.uuid4().hex
    for old_id, new_id in mapping.items():
        if old_id == new_id:
            continue
        old_name = _agent_name(old_id)
        new_name = _agent_name(new_id)
        temp_agent = f"tmp:rekey:agent:{agent_uuid}:{old_id}"
        if r.exists(old_name):
            if r.exists(temp_agent):
                r.delete(temp_agent)
            r.rename(old_name, temp_agent)
        if r.sismember("agents", old_name):
            r.srem("agents", old_name)
            r.sadd("agents", temp_agent)
        convs = _collect_conversations_for_agent(old_name)
        for ck in convs:
            try:
                _, phone = ck.split(":", 1)
            except ValueError:
                continue
            old_msg = _convo_msg_key(old_name, phone)
            temp_msg = _convo_msg_key(temp_agent, phone)
            if r.exists(old_msg):
                if r.exists(temp_msg):
                    r.delete(temp_msg)
                r.rename(old_msg, temp_msg)
            old_meta = _convo_meta_key(old_name, phone)
            temp_meta = _convo_meta_key(temp_agent, phone)
            if r.exists(old_meta):
                if r.exists(temp_meta):
                    r.delete(temp_meta)
                r.rename(old_meta, temp_meta)
                try:
                    r.hset(temp_meta, "agent_id", temp_agent)
                except Exception:
                    pass
            r.srem("conversations", ck)
            r.sadd("conversations", f"{temp_agent}:{phone}")
        if r.exists(temp_agent):
            if r.exists(new_name):
                r.delete(new_name)
            r.rename(temp_agent, new_name)
        if r.sismember("agents", temp_agent):
            r.srem("agents", temp_agent)
            r.sadd("agents", new_name)
        convs_temp = _collect_conversations_for_agent(temp_agent)
        for tck in convs_temp:
            try:
                _, phone = tck.split(":", 1)
            except ValueError:
                continue
            temp_msg = _convo_msg_key(temp_agent, phone)
            final_msg = _convo_msg_key(new_name, phone)
            if r.exists(temp_msg):
                if r.exists(final_msg):
                    r.delete(final_msg)
                r.rename(temp_msg, final_msg)
            temp_meta = _convo_meta_key(temp_agent, phone)
            final_meta = _convo_meta_key(new_name, phone)
            if r.exists(temp_meta):
                if r.exists(final_meta):
                    r.delete(final_meta)
                r.rename(temp_meta, final_meta)
                try:
                    r.hset(final_meta, "agent_id", new_name)
                except Exception:
                    pass
            if r.sismember("conversations", tck):
                r.srem("conversations", tck)
                r.sadd("conversations", f"{new_name}:{phone}")
    r.delete("agents")
    final_agent_ids = []
    for new_idx in sorted(mapping.values()):
        name = _agent_name(new_idx)
        if r.exists(name):
            r.sadd("agents", name)
            final_agent_ids.append(new_idx)
    r.set("next_agent_id", len(final_agent_ids))
    return {"status": "ok", "agents_before": len(ids), "agents_after": len(final_agent_ids)}

# ----------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------
@app.get("/")
@app.get("/")
def health():
    redis_status = "Failed"
    redis_error = ""
    if r:
        try:
            r.ping()
            redis_status = "Connected"
        except Exception as e:
            redis_status = "Failed"
            redis_error = str(e)
    else:
        redis_error = "Redis client not initialized"
    return {
        "message": "SMS Runtime Backend Live!",
        "redis": redis_status,
        "redis_error": redis_error,
        "endpoints": ["/add", "/fetch", "/update", "/delete", "/admin/compact"]
    }

# ----------------------------------------------------------------------
# TABLE CREATE/DELETE ENDPOINTS
# ----------------------------------------------------------------------
@app.post("/createtable")
def createtable(req: TableRequest):
    _require_redis()
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Table name required")
    _create_table(name)
    return {"status": "success", "table": name}

@app.delete("/deletetable")
def deletetable(name: str):
    _require_redis()
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Table name required")
    _delete_table(name)
    return {"status": "deleted", "table": name}

# ----------------------------------------------------------------------
# Function management module (plug-and-play)
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

LOCAL_FUNC_DIR = Path("./local_functions")
LOCAL_FUNC_DIR.mkdir(parents=True, exist_ok=True)

# simple unsafe token blacklist (adjust as needed)
def _sanitize_code(code: str) -> str:
    # Pass through without restriction
    return code


def _run_code(code: str, inputs: dict) -> dict:
    """
    Unrestricted runner.
    This executes user-provided code with normal Python semantics.
    Inputs are injected into the execution environment under their keys.
    The executed code may perform imports, filesystem operations, networking, and other system calls.
    Use only in trusted environments.
    """
    env = {
        "__name__": "__main__",
        "__file__": "<dynamic_function>",
        **inputs
    }

    try:
        # Compile to get clearer tracebacks
        compiled = compile(code, "<dynamic_function>", "exec")
        exec(compiled, env)
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

    if "result" in env:
        try:
            # Ensure the result is JSON serializable
            json.dumps(env["result"])
            return env["result"]
        except Exception:
            return {"error": "Function returned a non-JSON-serializable value", "value_repr": repr(env["result"])}
    return {"error": "No result returned from function"}

def _save_local_function(fid: str, code: str, meta: dict):
    (LOCAL_FUNC_DIR / f"{fid}.py").write_text(code)
    (LOCAL_FUNC_DIR / f"{fid}.meta.json").write_text(json.dumps(meta))

def _load_local_function(fid: str):
    code = None
    meta = {}
    code_path = LOCAL_FUNC_DIR / f"{fid}.py"
    meta_path = LOCAL_FUNC_DIR / f"{fid}.meta.json"
    if code_path.exists():
        code = code_path.read_text()
    try:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
    except Exception:
        meta = {}
    return code, meta
def _list_all_functions() -> List[Dict[str, Any]]:
    functions = {}
    
    # -------------------------
    # Redis-based functions
    # -------------------------
    if r:
        try:
            for key in r.scan_iter(match="function:*:meta"):
                fid = key.split(":")[1]
                meta = r.hgetall(key) or {}

                functions[fid] = {
                    "id": fid,
                    "name": meta.get("name", ""),
                    "created_at": int(meta.get("created_at") or meta.get("ts") or 0),
                    "updated_at": int(meta.get("updated_at") or meta.get("ts") or 0),
                    "source": "redis"
                }
        except Exception:
            pass

    # -------------------------
    # Local filesystem functions
    # -------------------------
    try:
        for meta_file in LOCAL_FUNC_DIR.glob("*.meta.json"):
            fid = meta_file.stem.replace(".meta", "")
            if fid in functions:
                continue

            try:
                meta = json.loads(meta_file.read_text())
            except Exception:
                meta = {}

            functions[fid] = {
                "id": fid,
                "name": meta.get("name", ""),
                "created_at": int(meta.get("created_at") or meta.get("ts") or 0),
                "updated_at": int(meta.get("updated_at") or meta.get("ts") or 0),
                "source": "local"
            }
    except Exception:
        pass

    return sorted(
        functions.values(),
        key=lambda x: x["created_at"],
        reverse=True
    )

def _find_project_file(filename: str) -> Optional[Path]:
    # Normalize to a .py filename
    if not filename.endswith(".py"):
        filename = f"{filename}.py"
    try:
        for p in BASE_DIR.rglob(filename):
            if p.name == filename:
                return p
    except Exception as e:
        print(f"Error scanning project files: {e}")
    return None

@app.post("/createfunction")
async def create_function(file: UploadFile = File(...), name: str = Form(...)):
    content = await file.read()
    try:
        code = _sanitize_code(content.decode())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    fid = f"func{int(time.time())}"
    meta = {"name": name, "ts": int(time.time())}
    if r:
        try:
            r.set(f"function:{fid}:code", code)
            r.hset(f"function:{fid}:meta", mapping=meta)
            return {"id": fid, "status": "created"}
        except Exception:
            # fallback to local
            _save_local_function(fid, code, meta)
            return {"id": fid, "status": "local"}
    _save_local_function(fid, code, meta)
    return {"id": fid, "status": "local"}

@app.post("/runfunction")
async def run_function(request: Request, id: Optional[str] = None):
    # Parse body, but be tolerant if body is missing or not JSON
    try:
        body = await request.json()
    except Exception:
        body = {}

    if not isinstance(body, dict):
        body = {}

    # Primary: get function id from body["id"]
    fid = body.get("id")

    # Secondary: if no id in body, use query parameter ?id=...
    if not fid:
        fid = id

    if not fid:
        raise HTTPException(status_code=400, detail="Function id required")

    # Primary: if body.input exists and is a non-empty dict, use that
    inputs: Dict[str, Any] = {}
    input_obj = body.get("input")
    if isinstance(input_obj, dict) and len(input_obj) > 0:
        inputs = input_obj
    else:
        # Secondary: use entire body as input, optionally dropping "id"
        inputs = {k: v for k, v in body.items() if k != "id"}

    code = None
    # Try Redis first
    if r:
        try:
            code = r.get(f"function:{fid}:code")
        except Exception:
            code = None
    if not code:
        code, _ = _load_local_function(fid)
    if not code:
        return JSONResponse(status_code=404, content={"error": "Function not found"})

    result = _run_code(code, inputs)
    return {"result": result}
@app.get("/functions")
def list_functions():
    """
    Returns all custom functions with metadata.
    """
    return {
        "functions": _list_all_functions(),
        "count": len(_list_all_functions())
    }

@app.post("/updatefunction")
async def update_function(id: str = Form(...), file: UploadFile = File(...), name: str = Form(...)):
    content = await file.read()
    try:
        code = _sanitize_code(content.decode())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    meta = {"name": name, "ts": int(time.time())}
    if r:
        try:
            r.set(f"function:{id}:code", code)
            r.hset(f"function:{id}:meta", mapping=meta)
            return {"id": id, "status": "updated"}
        except Exception:
            _save_local_function(id, code, meta)
            return {"id": id, "status": "updated_local"}
    _save_local_function(id, code, meta)
    return {"id": id, "status": "updated_local"}

@app.delete("/deletefunction")
def delete_function(id: str):
    deleted = False
    if r:
        try:
            r.delete(f"function:{id}:code")
            r.delete(f"function:{id}:meta")
            deleted = True
        except Exception:
            pass
    try:
        (LOCAL_FUNC_DIR / f"{id}.py").unlink(missing_ok=True)
        (LOCAL_FUNC_DIR / f"{id}.meta.json").unlink(missing_ok=True)
        deleted = True
    except Exception:
        pass
    if deleted:
        return {"id": id, "status": "deleted"}
    raise HTTPException(status_code=404, detail="Function not found")

# ----------------------------------------------------------------------
# NEW: /int endpoint to run project Python files by name
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# FINAL /int endpoint — BLOCKING for all files EXCEPT the hourly A2P job
# ----------------------------------------------------------------------
import sys
from io import StringIO

@app.api_route("/int", methods=["GET", "POST"])
async def int_endpoint(request: Request, file: str):
    target = _find_project_file(file)
    if not target:
        raise HTTPException(status_code=404, detail="File not found")

    # === Extract payload/input exactly like your old injection logic ===
    try:
        body = await request.json()
    except:
        body = {}

    payload = body.get("payload", {})
    input_data = body.get("input", {})
    combined_input = {}
    if isinstance(payload, dict):
        combined_input.update(payload)
    if isinstance(input_data, dict):
        combined_input.update(input_data)

    # === ONLY hourly A2P job runs in background ===
    if target.stem == "a2p-profile-every-hour-check-runtime":
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        def runner():
            try:
                code = target.read_text()
                env = {"__name__": "__main__", "__file__": str(target)}
                exec(compile(code, str(target), "exec"), env)
            except Exception:
                traceback.print_exc()
        threading.Thread(target=runner, daemon=True).start()
        return {
            "status": "started (background)",
            "file": target.name,
            "job_id": job_id,
            "note": "Hourly A2P check running"
        }

    # === ALL OTHER FILES: Full blocking execution with result ===
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = StringIO()
    sys.stdout = redirected_output
    sys.stderr = redirected_output

    try:
        code = target.read_text()
        env = {
            "__name__": "__main__",
            "__file__": str(target),
            "inputs": combined_input,        # ← now defined!
            "payload": payload,
            "input": input_data
        }
        exec(compile(code, str(target), "exec"), env)

        result = env.get("result")
        if result is not None:
            return {"result": result, "output": redirected_output.getvalue().strip()}

        output = redirected_output.getvalue().strip()
        return {
            "status": "completed",
            "file": target.name,
            "output": output or "No output"
        }

    except Exception as e:
        error_detail = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "output": redirected_output.getvalue().strip()
        }
        raise HTTPException(status_code=500, detail=error_detail)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# ----------------------------------------------------------------------
# ADD
# ----------------------------------------------------------------------
@app.post("/add")
def add_endpoint(req: AddRequest):
    _require_redis()
    data = req.data.copy()
    ts = int(time.time())
    data["created_at"] = ts
    data["updated_at"] = ts

    # --- Dynamic Table Support ---
    if req.table not in ["users", "agents", "conversations"]:
        if not _table_exists(req.table):
            raise HTTPException(status_code=404, detail="Table does not exist")
        ids_key = _table_ids_key(req.table)
        next_id_key = f"nextid:{req.table}"
        next_id = int(r.get(next_id_key) or 0) + 1
        r.set(next_id_key, next_id)
        row_key = _table_row_key(req.table, next_id)
        r.hset(row_key, mapping=data)
        r.sadd(ids_key, next_id)
        return {"id": str(next_id), "table": req.table, "status": "success"}

    if req.table == "users":
        nxt = _get_next_user_id() + 1
        r.set("next_user_id", nxt)
        uid = nxt
        r.hset(_user_key(uid), mapping=data)
        r.sadd("users", str(uid))
        return {"id": str(uid), "status": "success"}

    if req.table == "agents":
        nxt = _get_next_agent_id() + 1
        r.set("next_agent_id", nxt)
        aid_name = _agent_name(nxt)
        if "tools" in data:
            data["tools"] = json.dumps(data["tools"])
        r.hset(aid_name, mapping=data)
        r.sadd("agents", aid_name)
        return {"id": aid_name, "status": "success"}

    if req.table == "conversations":
        agent_id = data.get("agent_id")
        phone = data.get("phone")
        if not agent_id or not phone:
            raise HTTPException(status_code=400, detail="agent_id and phone required")
        key = _convo_msg_key(agent_id, phone)
        meta_key = _convo_meta_key(agent_id, phone)
        messages = data.pop("messages", [])
        r.delete(key)
        for m in messages:
            r.rpush(key, json.dumps(m))
        r.expire(key, 30 * 86400)
        r.hset(meta_key, mapping=data)
        r.sadd("conversations", f"{agent_id}:{phone}")
        return {"id": f"{agent_id}:{phone}", "status": "success"}

    raise HTTPException(status_code=400, detail="Invalid table")

# ----------------------------------------------------------------------
# FETCH - optimized with Redis pipelines
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

    # --- Dynamic Table Fetch ---
    if table not in ["users", "agents", "conversations"]:
        if not _table_exists(table):
            raise HTTPException(status_code=404, detail="Table does not exist")
        ids_key = _table_ids_key(table)
        ids = r.smembers(ids_key)
        out = {}
        if id:
            row_key = _table_row_key(table, id)
            rec = r.hgetall(row_key)
            return rec if rec else {}
        for row_id in ids:
            if row_id == "_meta":
                continue
            row_key = _table_row_key(table, row_id)
            rec = r.hgetall(row_key)
            if rec:
                out[row_id] = rec
        return out

    # USERS - batch HGETALL via pipeline
    if table == "users":
        if id:
            try:
                uid = int(id)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid user id")
            rec = dict(r.hgetall(_user_key(uid)))
            if not rec:
                return {}
            rec.setdefault("name", "")
            rec.setdefault("phone", "")
            rec["last_active"] = int(rec.get("last_active", 0))
            rec["created_at"] = int(rec.get("created_at", 0))
            return rec if _matches(rec, filt) else {}
        max_id = _get_next_user_id()
        out: Dict[str, Any] = {}
        if max_id > 0 and max_id <= MAX_FETCH_KEYS:
            keys = [_user_key(uid) for uid in range(1, max_id + 1)]
        else:
            members = r.smembers("users")
            keys = [_user_key(int(x)) for x in sorted([int(x) for x in members])]
        if not keys:
            return out
        pipe = r.pipeline()
        for k in keys:
            pipe.hgetall(k)
        results = pipe.execute()
        for uid, rec in zip([int(k.split(":")[1]) for k in keys], results):
            if rec:
                rec.setdefault("name", "")
                rec.setdefault("phone", "")
                rec["last_active"] = int(rec.get("last_active", 0))
                rec["created_at"] = int(rec.get("created_at", 0))
                if _matches(rec, filt):
                    out[str(uid)] = rec
        return out

    # AGENTS - batch HGETALL via pipeline
    if table == "agents":
        if id:
            aid = id if id.startswith("agent") else _agent_name(int(id))
            rec = dict(r.hgetall(aid))
            if not rec:
                return {}
            rec.setdefault("prompt", "")
            rec.setdefault("user_id", "")
            rec["tools"] = json.loads(rec["tools"]) if "tools" in rec and rec["tools"] else []
            rec["created_at"] = int(rec.get("created_at", 0))
            rec["updated_at"] = int(rec.get("updated_at", 0))
            return rec if _matches(rec, filt) else {}
        max_agent = _get_next_agent_id()
        out: Dict[str, Any] = {}
        if max_agent > 0 and max_agent <= MAX_FETCH_KEYS:
            agent_keys = [_agent_name(n) for n in range(1, max_agent + 1)]
        else:
            members = r.smembers("agents")
            agent_keys = sorted(list(members))
        if not agent_keys:
            return out
        pipe = r.pipeline()
        for aid in agent_keys:
            pipe.hgetall(aid)
        results = pipe.execute()
        for aid, rec in zip(agent_keys, results):
            if rec:
                rec.setdefault("prompt", "")
                rec.setdefault("user_id", "")
                rec["tools"] = json.loads(rec["tools"]) if "tools" in rec and rec["tools"] else []
                rec["created_at"] = int(rec.get("created_at", 0))
                rec["updated_at"] = int(rec.get("updated_at", 0))
                if _matches(rec, filt):
                    out[aid] = rec
        return out

    # CONVERSATIONS - batch LRANGE and HGETALL via pipeline
    if table == "conversations":
        if id:
            try:
                agent_id, phone = id.split(":", 1)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid conversation id")
            key = _convo_msg_key(agent_id, phone)
            msgs = []
            if r.exists(key):
                msgs = [json.loads(m) for m in r.lrange(key, 0, -1)]
            meta_key = _convo_meta_key(agent_id, phone)
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
        out: Dict[str, Any] = {}
        convs = list(r.smembers("conversations"))
        if not convs:
            return out
        pipe = r.pipeline()
        for ck in convs:
            try:
                agent_id, phone = ck.split(":", 1)
            except ValueError:
                continue
            msg_key = _convo_msg_key(agent_id, phone)
            meta_key = _convo_meta_key(agent_id, phone)
            pipe.lrange(msg_key, 0, -1)
            pipe.hgetall(meta_key)
        results = pipe.execute()
        it = iter(results)
        for ck in convs:
            try:
                agent_id, phone = ck.split(":", 1)
            except ValueError:
                continue
            try:
                msgs_raw = next(it)
                meta_raw = next(it)
            except StopIteration:
                break
            msgs = [json.loads(m) for m in msgs_raw] if msgs_raw else []
            meta = meta_raw or {}
            meta.setdefault("agent_id", agent_id)
            meta.setdefault("phone", phone)
            meta["created_at"] = int(meta.get("created_at", 0))
            meta["updated_at"] = int(meta.get("updated_at", 0))
            if not filt or any(_matches(m, filt) for m in msgs):
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

    # --- Dynamic Table Update ---
    if req.table not in ["users", "agents", "conversations"]:
        if not _table_exists(req.table):
            raise HTTPException(status_code=404, detail="Table does not exist")
        row_key = _table_row_key(req.table, req.id)
        if not r.exists(row_key):
            raise HTTPException(status_code=404, detail="Row not found")
        r.hset(row_key, mapping=updates)
        return {"status": "success", "id": req.id}

    if req.table == "users":
        try:
            uid = int(req.id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid user id")
        key = _user_key(uid)
        if not r.exists(key):
            raise HTTPException(status_code=404, detail="User not found")
        r.hset(key, mapping=updates)
        return {"status": "success"}

    if req.table == "agents":
        aid = req.id if req.id.startswith("agent") else _agent_name(int(req.id))
        if not r.exists(aid):
            raise HTTPException(status_code=404, detail="Agent not found")
        if "tools" in updates:
            updates["tools"] = json.dumps(updates["tools"])
        r.hset(aid, mapping=updates)
        return {"status": "success"}

    if req.table == "conversations":
        try:
            agent_id, phone = req.id.split(":", 1)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid conversation id")
        meta_key = _convo_meta_key(agent_id, phone)
        msg_key = _convo_msg_key(agent_id, phone)
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
# DELETE with automatic compaction
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# DELETE ALL RECORDS OF A DYNAMIC TABLE (but keep the table itself)
# ----------------------------------------------------------------------
@app.delete("/table/clear")
def clear_dynamic_table(name: str):
    """
    Deletes ALL rows inside a dynamic table but keeps the table definition.
    Does NOT remove the table from the global 'tables' set.
    Example:
        DELETE /table/clear?name=log
    """
    _require_redis()
    name = name.strip()

    if not name:
        raise HTTPException(status_code=400, detail="Table name required")

    if name in ["users", "agents", "conversations"]:
        raise HTTPException(status_code=400, detail="Cannot clear system tables")

    # Validate table exists
    if not _table_exists(name):
        raise HTTPException(status_code=404, detail="Table does not exist")

    ids_key = _table_ids_key(name)
    row_ids = r.smembers(ids_key)

    # Delete all rows
    deleted_count = 0
    for rid in row_ids:
        if rid == "_meta":
            continue
        r.delete(_table_row_key(name, rid))
        deleted_count += 1

    # Reset table id set and nextid counter
    r.delete(ids_key)
    r.sadd(ids_key, "_meta")
    r.set(f"nextid:{name}", 0)

    # Optionally clear meta:
    r.delete(_table_meta_key(name))
    r.hset(_table_meta_key(name), mapping={"created_at": int(time.time())})

    return {
        "status": "success",
        "table": name,
        "deleted_rows": deleted_count,
        "message": f"All rows cleared for table '{name}'"
    }
@app.delete("/table/clear")
def clear_table(name: str):
    """
    Deletes ALL records inside a dynamic table but keeps the table definition.
    Example:
        DELETE /table/clear?name=logs
    """
    _require_redis()

    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Table name required")

    # Prevent clearing system tables
    if name in ["users", "agents", "conversations"]:
        raise HTTPException(
            status_code=400,
            detail="System tables cannot be cleared"
        )

    # Ensure table exists
    if not _table_exists(name):
        raise HTTPException(status_code=404, detail="Table does not exist")

    ids_key = _table_ids_key(name)
    row_ids = r.smembers(ids_key)

    deleted = 0
    for rid in row_ids:
        if rid == "_meta":
            continue
        r.delete(_table_row_key(name, rid))
        deleted += 1

    # Reset ID tracking
    r.delete(ids_key)
    r.sadd(ids_key, "_meta")
    r.set(f"nextid:{name}", 0)

    # Reset table meta timestamp
    r.delete(_table_meta_key(name))
    r.hset(
        _table_meta_key(name),
        mapping={"created_at": int(time.time())}
    )

    return {
        "status": "success",
        "table": name,
        "deleted_rows": deleted
    }
@app.post("/table/bulk-delete")
def bulk_delete_rows(name: str, req: BulkDeleteRequest):
    _require_redis()

    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Table name required")

    if name in ["users", "agents", "conversations"]:
        raise HTTPException(
            status_code=400,
            detail="Bulk delete not allowed on system tables"
        )

    if not _table_exists(name):
        raise HTTPException(status_code=404, detail="Table does not exist")

    ids_key = _table_ids_key(name)

    deleted = []
    skipped = []

    for rid in req.ids:
        rid = str(rid)
        row_key = _table_row_key(name, rid)

        if r.exists(row_key):
            r.delete(row_key)
            r.srem(ids_key, rid)
            deleted.append(rid)
        else:
            skipped.append(rid)

    return {
        "status": "success",
        "table": name,
        "deleted_ids": deleted,
        "skipped_ids": skipped,
        "deleted_count": len(deleted)
    }

@app.delete("/delete")
def delete_endpoint(table: str, id: str):
    _require_redis()

    # --- Dynamic Table Delete ---
    if table not in ["users", "agents", "conversations"]:
        if not _table_exists(table):
            raise HTTPException(status_code=404, detail="Table does not exist")
        ids_key = _table_ids_key(table)
        row_key = _table_row_key(table, id)
        if r.exists(row_key):
            r.delete(row_key)
            r.srem(ids_key, id)
            return {"status": "deleted", "id": id}
        raise HTTPException(status_code=404, detail="Row not found")

    if table == "users":
        try:
            uid = int(id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid user id")
        key = _user_key(uid)
        if r.delete(key):
            r.srem("users", str(uid))
            compact_users()
            return {"status": "deleted", "id": str(uid)}
        raise HTTPException(status_code=404, detail="User not found")

    if table == "agents":
        if id.startswith("agent"):
            try:
                idx = int(id[len("agent"):])
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid agent id")
        else:
            try:
                idx = int(id)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid agent id")
        aid_name = _agent_name(idx)
        if r.delete(aid_name):
            r.srem("agents", aid_name)
            compact_agents()
            return {"status": "deleted", "id": aid_name}
        raise HTTPException(status_code=404, detail="Agent not found")

    if table == "conversations":
        try:
            agent_id, phone = id.split(":", 1)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid conversation id")
        msg_key = _convo_msg_key(agent_id, phone)
        meta_key = _convo_meta_key(agent_id, phone)
        deleted = r.delete(msg_key, meta_key)
        if deleted:
            r.srem("conversations", id)
            return {"status": "deleted", "id": id}
        raise HTTPException(status_code=404, detail="Conversation not found")

    raise HTTPException(status_code=400, detail="Invalid table")

# ----------------------------------------------------------------------
# Admin compaction endpoint
# ----------------------------------------------------------------------
@app.post("/admin/compact")
def admin_compact(payload: Optional[Dict[str, Any]] = None):
    _require_redis()
    table = None
    if payload and isinstance(payload, dict):
        table = payload.get("table")
    result = {}
    if table is None or table == "users" or table == "all":
        result["users"] = compact_users()
    if table is None or table == "agents" or table == "all":
        result["agents"] = compact_agents()
    return {"status": "ok", "result": result}

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
