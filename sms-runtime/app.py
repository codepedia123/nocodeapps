# sms-runtime/app.py
import os
import json
import time
from typing import Dict, Any, Optional

import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----------------------------------------------------------------------
# Redis connection – uses Render's REDIS_URL or fallback KV instance
# ----------------------------------------------------------------------
redis_url = os.getenv("REDIS_URL", "redis://red-d44f17jipnbc73dqs2k0:6379")
try:
    r = redis.from_url(redis_url, decode_responses=True)
    r.ping()
except Exception as e:
    print(f"Redis connection failed: {e}")
    r = None

app = FastAPI(title="SMS Runtime Backend")

# allow any origin for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Request Models
# ----------------------------------------------------------------------
class AddRequest(BaseModel):
    table: str
    data: Dict[str, Any]

class UpdateRequest(BaseModel):
    table: str
    id: str
    updates: Dict[str, Any]

# ----------------------------------------------------------------------
# Helpers: ID management and shifting logic
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

# Shift user IDs down after deleting a user with numeric id `deleted_id`
def _shift_users_down(deleted_id: int):
    next_id = _get_next_user_id()
    if deleted_id >= next_id or deleted_id < 1:
        # nothing to shift
        return

    # Move each user i -> i-1 for i in deleted_id+1 .. next_id
    for i in range(deleted_id + 1, next_id + 1):
        old = _user_key(i)
        new = _user_key(i - 1)
        if r.exists(old):
            # ensure destination does not exist to avoid RENAME error
            if r.exists(new):
                r.delete(new)
            r.rename(old, new)
            # update membership set
            r.srem("users", str(i))
            r.sadd("users", str(i - 1))

    # Update agents that reference user ids
    # Agents are stored as hashes named "agent{n}"
    for aid in r.smembers("agents"):
        try:
            # read user_id if present
            user_id = r.hget(aid, "user_id")
            if user_id is None:
                continue
            # user_id stored as string number
            if user_id.isdigit():
                uid = int(user_id)
                if uid > deleted_id:
                    # decrement
                    r.hset(aid, "user_id", str(uid - 1))
        except Exception:
            continue

    # decrement next_user_id
    r.decr("next_user_id")

# Shift agent IDs down after deleting an agent with numeric index `deleted_index`
# This will rename agent hashes and update conversation keys/meta and the conversations set.
def _shift_agents_down(deleted_index: int):
    next_agent = _get_next_agent_id()
    if deleted_index >= next_agent or deleted_index < 1:
        return

    for j in range(deleted_index + 1, next_agent + 1):
        old_name = _agent_name(j)
        new_name = _agent_name(j - 1)

        # Rename agent hash (if exists)
        if r.exists(old_name):
            if r.exists(new_name):
                r.delete(new_name)
            r.rename(old_name, new_name)

        # Update 'agents' set membership
        r.srem("agents", old_name)
        r.sadd("agents", new_name)

        # Update any conversations that reference this agent
        # We must scan existing conversations set entries and replace the prefix.
        for ck in list(r.smembers("conversations")):
            # ck format: "agentN:+phone" or "agentN:phone"
            if not ck:
                continue
            if ck.startswith(old_name + ":"):
                try:
                    # split into agent portion and phone
                    _, phone = ck.split(":", 1)
                except ValueError:
                    continue
                old_msg_key = _convo_msg_key(old_name, phone)
                new_msg_key = _convo_msg_key(new_name, phone)
                old_meta_key = _convo_meta_key(old_name, phone)
                new_meta_key = _convo_meta_key(new_name, phone)

                # rename message list/key
                if r.exists(old_msg_key):
                    if r.exists(new_msg_key):
                        r.delete(new_msg_key)
                    r.rename(old_msg_key, new_msg_key)

                # rename meta hash and update its internal agent_id field if present
                if r.exists(old_meta_key):
                    if r.exists(new_meta_key):
                        r.delete(new_meta_key)
                    r.rename(old_meta_key, new_meta_key)
                    try:
                        r.hset(new_meta_key, "agent_id", new_name)
                    except Exception:
                        pass

                # update conversations set entry
                r.srem("conversations", ck)
                r.sadd("conversations", f"{new_name}:{phone}")

    # decrement next_agent_id
    r.decr("next_agent_id")

# ----------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------
@app.get("/")
def health():
    redis_status = "Connected" if r and r.ping() else "Failed"
    return {
        "message": "SMS Runtime Backend Live!",
        "redis": redis_status,
        "endpoints": ["/add", "/fetch", "/update", "/delete"]
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
        # replace any existing conversation (explicit behavior from original)
        r.delete(key)
        for m in messages:
            r.rpush(key, json.dumps(m))
        r.expire(key, 30 * 86400)
        r.hset(meta_key, mapping=data)
        r.sadd("conversations", f"{agent_id}:{phone}")
        return {"id": f"{agent_id}:{phone}", "status": "success"}

    raise HTTPException(status_code=400, detail="Invalid table")

# ----------------------------------------------------------------------
# FETCH
# Note: For deterministic ordering and speed we iterate numeric ids for users and agents.
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

    # USERS: iterate from 1..next_user_id to preserve order and keep fetch fast
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

        out = {}
        max_id = _get_next_user_id()
        for uid in range(1, max_id + 1):
            key = _user_key(uid)
            if not r.exists(key):
                continue
            rec = dict(r.hgetall(key))
            if rec and _matches(rec, filt):
                rec.setdefault("name", "")
                rec.setdefault("phone", "")
                rec["last_active"] = int(rec.get("last_active", 0))
                rec["created_at"] = int(rec.get("created_at", 0))
                out[str(uid)] = rec
        return out

    # AGENTS: ordered by numeric suffix
    if table == "agents":
        if id:
            # id might be "agent3" or "3"
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
        out = {}
        max_agent = _get_next_agent_id()
        for n in range(1, max_agent + 1):
            aid = _agent_name(n)
            if not r.exists(aid):
                continue
            rec = dict(r.hgetall(aid))
            if rec and _matches(rec, filt):
                rec.setdefault("prompt", "")
                rec.setdefault("user_id", "")
                rec["tools"] = json.loads(rec["tools"]) if "tools" in rec and rec["tools"] else []
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

        # return all conversations (careful: could be large)
        out = {}
        for ck in r.smembers("conversations"):
            if not ck:
                continue
            try:
                agent_id, phone = ck.split(":", 1)
            except ValueError:
                continue
            key = _convo_msg_key(agent_id, phone)
            msgs = [json.loads(m) for m in r.lrange(key, 0, -1)] if r.exists(key) else []
            meta_key = _convo_meta_key(agent_id, phone)
            meta = dict(r.hgetall(meta_key)) if r.exists(meta_key) else {}
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
# UPDATE
# ----------------------------------------------------------------------
@app.post("/update")
def update_endpoint(req: UpdateRequest):
    _require_redis()
    ts = int(time.time())
    updates = req.updates.copy()
    updates["updated_at"] = ts

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
# DELETE with shifting behavior
# ----------------------------------------------------------------------
@app.delete("/delete")
def delete_endpoint(table: str, id: str):
    _require_redis()

    # USERS: id is numeric (string) -> delete and shift down remaining users
    if table == "users":
        try:
            uid = int(id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid user id")
        key = _user_key(uid)
        if r.delete(key):
            r.srem("users", str(uid))
            # Shift later users down so positions remain contiguous
            _shift_users_down(uid)
            return {"status": "deleted", "id": str(uid)}
        raise HTTPException(status_code=404, detail="User not found")

    # AGENTS: id may be "agentN" or numeric "N"
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
            # Shift later agents down
            _shift_agents_down(idx)
            return {"status": "deleted", "id": aid_name}
        raise HTTPException(status_code=404, detail="Agent not found")

    # CONVERSATIONS: delete specific conversation by id agent:phone
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
# Run (use uvicorn sms-runtime.app:app)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
