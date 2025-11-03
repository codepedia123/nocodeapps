import os
from typing import Optional, Dict, Any, Union
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import redis
import json
from pydantic import BaseModel
import time

# Redis connection (Render env or local fallback)
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
try:
    r = redis.from_url(redis_url, decode_responses=True)
    r.ping()  # Test connection
except redis.exceptions.ConnectionError as e:
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

class AddRequest(BaseModel):
    table: str
    data: Dict[str, Any]

class UpdateRequest(BaseModel):
    table: str
    id: str
    updates: Dict[str, Any]

class FetchRequest(BaseModel):
    table: str
    id: Optional[str] = None
    filters: Optional[str] = None

# Embedded CRUD Functions (Self-Contained - No External Imports)
def add_record(table: str, data: Dict[str, Any]) -> str:
    """Add new record to Users, Agents, or Conversations with auto-ID and timestamps."""
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

def fetch_record(table: str, id_key: Optional[str] = None, filter_str: Optional[str] = None) -> Union[Dict, list]:
    if not r:
        return {"error": "Redis not connected"}
    
    filters = parse_filters(filter_str)

    if table == "users":
        if id_key:
            record = dict(r.hgetall(f"user:{id_key}"))
            if not record:
                return {}
            record = ensure_timestamps(record)
            return record if (not filters or filter_record(record, filters)) else {}
        all_ids = r.smembers("users")
        users = {}
        for uid in all_ids:
            rec = dict(r.hgetall(f"user:{uid}"))
            if rec:
                rec = ensure_timestamps(rec)
                if not filters or filter_record(rec, filters):
                    users[uid] = rec
        return users or {}

    elif table == "agents":
        if id_key:
            record = dict(r.hgetall(id_key))
            if not record:
                return {}
            if "tools" in record:
                record["tools"] = json.loads(record["tools"])
            record = ensure_timestamps(record)
            return record if (not filters or filter_record(record, filters)) else {}
        all_ids = r.smembers("agents")
        agents = {}
        for aid in all_ids:
            rec = dict(r.hgetall(aid))
            if rec:
                if "tools" in rec:
                    rec["tools"] = json.loads(rec["tools"])
                rec = ensure_timestamps(rec)
                if not filters or filter_record(rec, filters):
                    agents[aid] = rec
        return agents or {}

    elif table == "conversations":
        if id_key:
            agent_id, phone = id_key.split(":", 1)
            key = f"convo:{agent_id}:{phone}"
            meta_key = f"convo_meta:{agent_id}:{phone}"
            messages = [json.loads(m) for m in r.lrange(key, 0, -1)]
            meta = dict(r.hgetall(meta_key))
            if not messages and not meta:
                return {"messages": [], "meta": {}}
            meta = ensure_timestamps(meta)
            if filters:
                filtered = [m for m in messages if filter_record(m, filters)]
                if not filtered:
                    return {"messages": [], "meta": meta}
                return {"messages": filtered, "meta": meta}
            return {"messages": messages, "meta": meta}

        all_keys = r.smembers("conversations")
        convos = {}
        for ck in all_keys:
            agent_id, phone = ck.split(":", 1)
            key = f"convo:{agent_id}:{phone}"
            meta_key = f"convo_meta:{agent_id}:{phone}"
            messages = [json.loads(m) for m in r.lrange(key, 0, -1)]
            meta = dict(r.hgetall(meta_key))
            if messages or meta:
                meta = ensure_timestamps(meta)
                if filters:
                    if any(filter_record(m, filters) for m in messages):
                        convos[ck] = {"messages": messages, "meta": meta}
                else:
                    convos[ck] = {"messages": messages, "meta": meta}
        return convos or {}

    return {"error": f"Invalid table: {table}"}

def parse_filters(filter_str: Optional[str]) -> Dict[str, str]:
    if not filter_str:
        return {}
    filters = {}
    for pair in filter_str.split(","):
        pair = pair.strip()
        if '=' in pair:
            field, value = pair.split('=', 1)
            filters[field.strip()] = value.strip()
    return filters

def filter_record(record: Dict[str, Any], filters: Dict[str, str]) -> bool:
    for field, value in filters.items():
        if record.get(field) != value:
            return False
    return True

def ensure_timestamps(record: Dict[str, Any]) -> Dict[str, Any]:
    record["created_at"] = int(record.get("created_at", 0))
    record["updated_at"] = int(record.get("updated_at", 0))
    return record

# --- UPDATE ---
def update_record(table: str, id_key: str, updates: Dict[str, Any]) -> bool:
    updates["updated_at"] = int(time.time())

    if table == "users":
        key = f"user:{id_key}"
        if not r.exists(key):
            return False
        r.hset(key, mapping=updates)
        return True

    elif table == "agents":
        key = id_key
        if not r.exists(key):
            return False
        if "tools" in updates:
            updates["tools"] = json.dumps(updates["tools"])
        r.hset(key, mapping=updates)
        return True

    elif table == "conversations":
        agent_id, phone = id_key.split(":", 1)
        key = f"convo:{agent_id}:{phone}"
        meta_key = f"convo_meta:{agent_id}:{phone}"
        if not r.exists(key):
            return False
        if "append" in updates:
            for msg in updates["append"]:
                r.rpush(key, json.dumps(msg))
            r.expire(key, 86400 * 30)
            r.hset(meta_key, "updated_at", updates["updated_at"])
            return True
        else:
            r.hset(meta_key, mapping=updates)
            return True

    return False

# --- ROUTES ---
@app.post("/add")
def add_endpoint(request: AddRequest):
    try:
        result_id = add_record(request.table, request.data)
        return {"id": result_id, "status": "success"}
    except Exception as e:
        return {"error": str(e)}, 400

@app.get("/fetch")
def fetch_endpoint(table: str, id_key: Optional[str] = None, filters: Optional[str] = None):
    try:
        result = fetch_record(table, id_key, filters)
        return result
    except Exception as e:
        return {"error": str(e)}, 400

@app.post("/update")
def update_endpoint(request: UpdateRequest):
    try:
        success = update_record(request.table, request.id, request.updates)
        if success:
            return {"status": "success"}
        else:
            return {"error": "Record not found"}, 404
    except Exception as e:
        return {"error": str(e)}, 400

@app.delete("/delete")
def delete_endpoint(table: str, id: str):
    try:
        success = delete_record(table, id)
        if success:
            return {"status": "deleted"}
        else:
            return {"error": "Record not found"}, 404
    except Exception as e:
        return {"error": str(e)}, 400

@app.get("/")
def root():
    try:
        if r and r.ping():
            return {"message": "SMS Runtime Backend Live on Render!", "redis": "Connected"}
        else:
            return {"message": "Backend Live, but Redis down - check connection."}, 503
    except Exception as e:
        return {"error": str(e)}, 500

@app.get("/users")
def get_users():
    try:
        if not r:
            raise ValueError("Redis not connected")
        all_ids = r.smembers("users")
        users = {uid: dict(r.hgetall(f"user:{uid}")) for uid in all_ids}
        return users
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
