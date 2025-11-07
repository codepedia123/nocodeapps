# sms-runtime/app.py
import os
import json
import time
import uuid
from typing import Dict, Any, Optional, List, Tuple

import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----------------------------------------------------------------------
# Redis connection – uses REDIS_URL or fallback
# ----------------------------------------------------------------------
redis_url = os.getenv("REDIS_URL", "redis://red-d44f17jipnbc73dqs2k0:6379")
try:
    r = redis.from_url(redis_url, decode_responses=True)
    r.ping()
except Exception as e:
    print(f"Redis connection failed: {e}")
    r = None

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

class UpdateRequest(BaseModel):
    table: str
    id: str
    updates: Dict[str, Any]

# ----------------------------------------------------------------------
# Helpers and key utilities
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

# ----------------------------------------------------------------------
# Compaction helpers
# These functions create contiguous ids and update dependent references.
# They use temporary keys to avoid rename collisions.
# ----------------------------------------------------------------------
def _all_numeric_suffix_ids(prefix: str) -> List[int]:
    # prefix examples: "user" or "agent"
    ids = []
    pattern = f"{prefix}:*"
    for key in r.scan_iter(match=pattern):
        parts = key.split(":")
        if len(parts) >= 2:
            suffix = parts[1]
            if suffix.isdigit():
                ids.append(int(suffix))
    ids = sorted(set(ids))
    return ids

def compact_users() -> Dict[str, Any]:
    """
    Make user ids sequential from 1..N. Update agents.user_id references.
    Return summary dict.
    """
    # gather current numeric user ids
    ids = _all_numeric_suffix_ids("user")
    if not ids:
        r.set("next_user_id", 0)
        r.delete("users")
        return {"status": "ok", "users_before": 0, "users_after": 0}

    # mapping old_id -> new_id
    mapping: Dict[int, int] = {}
    for new_idx, old_id in enumerate(ids, start=1):
        mapping[old_id] = new_idx

    # step 1: rename all old user keys that need renaming to temp keys
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

    # step 2: remove old numeric entries from set and add new ones for keys that remained (non-renamed)
    # rebuild users set cleanly later; first collect final ids with data
    final_ids: List[int] = []
    # keys that were not moved but are valid (old_id == new_id)
    for old_id, new_id in mapping.items():
        if old_id == new_id:
            key = _user_key(old_id)
            if r.exists(key):
                final_ids.append(new_id)

    # step 3: rename temps to final keys and record final_ids
    for temp_key, (old_id, new_id) in temp_map.items():
        final_key = _user_key(new_id)
        if r.exists(final_key):
            r.delete(final_key)
        r.rename(temp_key, final_key)
        final_ids.append(new_id)

    final_ids = sorted(set(final_ids))

    # step 4: rebuild users set to match final ids
    r.delete("users")
    for uid in final_ids:
        r.sadd("users", str(uid))

    # step 5: update agents.user_id fields that referenced old ids
    # map old numeric -> new numeric
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

    # step 6: set next_user_id correctly
    r.set("next_user_id", len(final_ids))
    return {"status": "ok", "users_before": len(ids), "users_after": len(final_ids)}

def _collect_conversations_for_agent(agent_name: str) -> List[str]:
    # returns conversation set entries like "agentN:phone" for this agent
    out = []
    for ck in r.smembers("conversations"):
        if not ck:
            continue
        if ck.startswith(agent_name + ":"):
            out.append(ck)
    return out

def compact_agents() -> Dict[str, Any]:
    """
    Make agent ids sequential from 1..N. Update conversation keys, meta, and the conversations set.
    Return summary dict.
    """
    ids = _all_numeric_suffix_ids("agent")
    if not ids:
        r.set("next_agent_id", 0)
        r.delete("agents")
        return {"status": "ok", "agents_before": 0, "agents_after": 0}

    mapping: Dict[int, int] = {}
    for new_idx, old_id in enumerate(ids, start=1):
        mapping[old_id] = new_idx

    agent_uuid = uuid.uuid4().hex

    # We'll perform per-agent renames using temporary agent names to avoid collisions.
    # For each old->new where different:
    for old_id, new_id in mapping.items():
        if old_id == new_id:
            continue
        old_name = _agent_name(old_id)
        new_name = _agent_name(new_id)
        temp_agent = f"tmp:rekey:agent:{agent_uuid}:{old_id}"

        # rename agent hash to temp
        if r.exists(old_name):
            if r.exists(temp_agent):
                r.delete(temp_agent)
            r.rename(old_name, temp_agent)

        # update agents set: remove old_name, add temp_agent placeholder
        if r.sismember("agents", old_name):
            r.srem("agents", old_name)
            r.sadd("agents", temp_agent)

        # rename conversation keys and meta that belong to old_name to temp_agent prefix
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
                # update agent_id field inside meta to temp_agent
                try:
                    r.hset(temp_meta, "agent_id", temp_agent)
                except Exception:
                    pass

            # update conversations set entry to temp_agent:phone
            r.srem("conversations", ck)
            r.sadd("conversations", f"{temp_agent}:{phone}")

        # finally rename temp_agent to final new_name
        if r.exists(temp_agent):
            if r.exists(new_name):
                r.delete(new_name)
            r.rename(temp_agent, new_name)

        # update agents set: remove temp_agent, add new_name
        if r.sismember("agents", temp_agent):
            r.srem("agents", temp_agent)
            r.sadd("agents", new_name)

        # rename any temp conversation entries we created to final names
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

            # update conversations set entry from temp_agent:phone to new_name:phone
            if r.sismember("conversations", tck):
                r.srem("conversations", tck)
                r.sadd("conversations", f"{new_name}:{phone}")

    # rebuild agents set so it contains only agentN names that actually exist
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
# Health endpoint
# ----------------------------------------------------------------------
@app.get("/")
def health():
    redis_status = "Connected" if r and r.ping() else "Failed"
    return {
        "message": "SMS Runtime Backend Live!",
        "redis": redis_status,
        "endpoints": ["/add", "/fetch", "/update", "/delete", "/admin/compact"]
    }

# ----------------------------------------------------------------------
# ADD endpoint
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
        r.delete(key)
        for m in messages:
            r.rpush(key, json.dumps(m))
        r.expire(key, 30 * 86400)
        r.hset(meta_key, mapping=data)
        r.sadd("conversations", f"{agent_id}:{phone}")
        return {"id": f"{agent_id}:{phone}", "status": "success"}

    raise HTTPException(status_code=400, detail="Invalid table")

# ----------------------------------------------------------------------
# FETCH endpoint
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

    # AGENTS
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
# UPDATE endpoint
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
# DELETE with automatic compaction
# ----------------------------------------------------------------------
@app.delete("/delete")
def delete_endpoint(table: str, id: str):
    _require_redis()

    # USERS
    if table == "users":
        try:
            uid = int(id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid user id")
        key = _user_key(uid)
        if r.delete(key):
            r.srem("users", str(uid))
            # compact to remove gaps and keep sequence contiguous
            compact_users()
            return {"status": "deleted", "id": str(uid)}
        raise HTTPException(status_code=404, detail="User not found")

    # AGENTS
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
            # compact agents and update conversation keys/meta
            compact_agents()
            return {"status": "deleted", "id": aid_name}
        raise HTTPException(status_code=404, detail="Agent not found")

    # CONVERSATIONS
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
# Admin - manual compaction endpoint
# Use POST /admin/compact with optional json body {"table": "users"|"agents"|"all"}
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
