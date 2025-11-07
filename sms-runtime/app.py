# sms-runtime/app.py
import os
import sys
import json
import time
from typing import Dict, Any, Optional, Union

import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----------------------------------------------------------------------
# Redis connection – Render supplies REDIS_URL; fallback to the KV instance
# ----------------------------------------------------------------------
redis_url = os.getenv("REDIS_URL", "redis://red-d44f17jipnbc73dqs2k0:6379")
try:
    r = redis.from_url(redis_url, decode_responses=True)
    r.ping()                     # confirm connectivity
except redis.ConnectionError as e:
    print(f"Redis connection failed: {e}")
    r = None                     # routes will raise a clear error

app = FastAPI(title="SMS Runtime Backend")

# CORS – allow any origin (tighten in prod if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Pydantic request models
# ----------------------------------------------------------------------
class AddRequest(BaseModel):
    table: str
    data: Dict[str, Any]

class UpdateRequest(BaseModel):
    table: str
    id: str
    updates: Dict[str, Any]

class DeleteRequest(BaseModel):
    table: str
    id: str

# ----------------------------------------------------------------------
# Helper: raise if Redis is unavailable
# ----------------------------------------------------------------------
def _require_redis():
    if not r:
        raise HTTPException(status_code=503, detail="Redis not available")

# ----------------------------------------------------------------------
# HEALTH CHECK
# ----------------------------------------------------------------------
@app.get("/")
def health():
    redis_status = "Connected" if r and r.ping() else "Failed"
    return {
        "message": "SMS Runtime Backend Live!",
        "redis": redis_status,
        "endpoints": ["/add", "/fetch", "/update", /delete"]
    }

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

    if req.table == "users":
        nxt = int(r.get("next_user_id") or 0) + 1
        r.set("next_user_id", nxt)
        uid = str(nxt)
        r.hset(f"user:{uid}", mapping=data)
        r.sadd("users", uid)
        return {"id": uid, "status": "success"}

    if req.table == "agents":
        nxt = int(r.get("next_agent_id") or 0) + 1
        r.set("next_agent_id", nxt)
        aid = f"agent{nxt}"
        if "tools" in data:
            data["tools"] = json.dumps(data["tools"])
        r.hset(aid, mapping=data)
        r.sadd("agents", aid)
        return {"id": aid, "status": "success"}

    if req.table == "conversations":
        agent_id = data.get("agent_id")
        phone = data.get("phone")
        if not agent_id or not phone:
            raise HTTPException(status_code=400, detail="agent_id and phone required")
        key = f"convo:{agent_id}:{phone}"
        meta_key = f"convo_meta:{agent_id}:{phone}"
        messages = data.pop("messages", [])
        r.delete(key)
        for m in messages:
            r.rpush(key, json.dumps(m))
        r.expire(key, 30 * 86400)               # 30 days
        r.hset(meta_key, mapping=data)
        r.sadd("conversations", f"{agent_id}:{phone}")
        return {"id": f"{agent_id}:{phone}", "status": "success"}

    raise HTTPException(status_code=400, detail="Invalid table")

# ----------------------------------------------------------------------
# FETCH (supports id + filters)
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

    # ---------- USERS ----------
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
        # all users
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

    # ---------- AGENTS ----------
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
        # all agents
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

    # ---------- CONVERSATIONS ----------
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
        # all conversations
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
# UPDATE (including message append)
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

        # Append new messages if `append` is supplied
        if "append" in updates:
            for m in updates.pop("append"):
                r.rpush(msg_key, json.dumps(m))
        # Update meta fields
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
# Entry point (Render uses `uvicorn sms-runtime.app:app`)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
