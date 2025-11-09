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
# Config
# ----------------------------------------------------------------------
redis_url = os.getenv("REDIS_URL", "redis://default:MCBSKQGovtRMYogRwmeZqAhIVGJ5@clustercfg.nocodeapps-redis.sm3cdo.use1.cache.amazonaws.com:6379")
# Maximum number of sequential ids to pipeline directly. If next_id is larger,
# we fall back to scanning the membership set to avoid huge pipelines.
MAX_FETCH_KEYS = int(os.getenv("MAX_FETCH_KEYS", "5000"))

# ----------------------------------------------------------------------
# Redis connection (safe, non-blocking)
# ----------------------------------------------------------------------
def _init_redis() -> Optional[redis.Redis]:
    """
    Initialize Redis client with short timeouts so import/startup does not hang
    if Redis is unreachable. Returns a connected client or None.
    """
    try:
        # set short connect and socket timeouts to avoid blocking import
        client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            health_check_interval=30
        )
    except Exception as e:
        print(f"Redis init failed: {e}")
        return None

    try:
        # attempt a quick ping to verify connectivity
        client.ping()
        print("✅ Redis connected.")
        return client
    except Exception as e:
        print(f"⚠️ Redis ping failed (will proceed with r=None): {e}")
        return None

# initialize once at import time but with safe timeouts
r = _init_redis()

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

    # rename to temp keys to avoid collisions
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
def health():
    redis_status = "Failed"
    if r:
        try:
            # ping using the configured socket timeout so it does not block long
            r.ping()
            redis_status = "Connected"
        except Exception:
            redis_status = "Failed"
    return {
        "message": "SMS Runtime Backend Live!",
        "redis": redis_status,
        "endpoints": ["/add", "/fetch", "/update", "/delete", "/admin/compact"]
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

        # If max_id is reasonable, pipeline sequential keys 1..max_id
        if max_id > 0 and max_id <= MAX_FETCH_KEYS:
            keys = [_user_key(uid) for uid in range(1, max_id + 1)]
        else:
            # fallback: iterate over set 'users' to get actual keys (keeps pipeline smaller)
            members = r.smembers("users")
            keys = [_user_key(int(x)) for x in sorted([int(x) for x in members])]

        if not keys:
            return out

        pipe = r.pipeline()
        for k in keys:
            pipe.hgetall(k)
        results = pipe.execute()

        # results correspond to keys; include only non-empty hgetall
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
        # for each conversation, queue LRANGE and HGETALL; store order
        for ck in convs:
            try:
                agent_id, phone = ck.split(":", 1)
            except ValueError:
                # skip malformed entries
                continue
            msg_key = _convo_msg_key(agent_id, phone)
            meta_key = _convo_meta_key(agent_id, phone)
            pipe.lrange(msg_key, 0, -1)
            pipe.hgetall(meta_key)

        results = pipe.execute()
        # results are in pairs [lrange1, hgetall1, lrange2, hgetall2, ...]
        it = iter(results)
        i = 0
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
            i += 1
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
# DELETE with automatic compaction
# ----------------------------------------------------------------------
@app.delete("/delete")
def delete_endpoint(table: str, id: str):
    _require_redis()

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
