import os
from typing import Dict, Any, Union, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import redis
import json
from pydantic import BaseModel

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

class DeleteRequest(BaseModel):
    table: str
    id: str

# Inline CRUD Functions (Self-Contained - No External Imports)
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

def fetch_record(table: str, id_key: Optional[str] = None, filter_str: Optional[str] = None) -> Union[Dict[str, Any], list, dict]:
    """Fetch all or specific record; supports filters (comma-separated 'field=value'). Returns full JSON."""
    if not r:
        raise ValueError("Redis not connected")
    
    # Parse filters (e.g., "name=John,email=john@example.com")
    filters = {}
    if filter_str:
        for pair in filter_str.split(','):
            pair = pair.strip()
            if '=' in pair:
                field, value = pair.split('=', 1)
                filters[field.strip()] = value.strip()
    
    def filter_record(record: Dict[str, Any], filters: Dict[str, str]) -> bool:
        for field, value in filters.items():
            if str(record.get(field, '')).lower() != value.lower():
                return False
        return True

    if table == "users":
        if id_key:
            record = dict(r.hgetall(f"user:{id_key}"))
            if not record:
                return {}
            # Add defaults for consistency
            record.setdefault("name", "")
            record.setdefault("phone", "")
            record["last_active"] = int(record.get("last_active", 0))
            record["created_at"] = int(record.get("created_at", 0))
            if filters and not filter_record(record, filters):
                return {}
            return record
        
        all_ids = r.smembers("users")
        users = {}
        for uid in all_ids:
            record = dict(r.hgetall(f"user:{uid}"))
            if record:
                record.setdefault("name", "")
                record.setdefault("phone", "")
                record["last_active"] = int(record.get("last_active", 0))
                record["created_at"] = int(record.get("created_at", 0))
                if not filters or filter_record(record, filters):
                    users[uid] = record
        return users if users else {}
    
    elif table == "agents":
        if id_key:
            record = dict(r.hgetall(id_key))  # Key is "agent1"
            if not record:
                return {}
            # Add defaults
            record.setdefault("prompt", "")
            record.setdefault("user_id", "")
            if "tools" in record:
                record["tools"] = json.loads(record["tools"])
            else:
                record["tools"] = []
            record["created_at"] = int(record.get("created_at", 0))
            record["updated_at"] = int(record.get("updated_at", 0))
            if filters and not filter_record(record, filters):
                return {}
            return record
        
        all_ids = r.smembers("agents")
        agents = {}
        for aid in all_ids:
            record = dict(r.hgetall(aid))
            if record:
                record.setdefault("prompt", "")
                record.setdefault("user_id", "")
                if "tools" in record:
                    record["tools"] = json.loads(record["tools"])
                else:
                    record["tools"] = []
                record["created_at"] = int(record.get("created_at", 0))
                record["updated_at"] = int(record.get("updated_at", 0))
                if not filters or filter_record(record, filters):
                    agents[aid] = record
        return agents if agents else {}
    
    elif table == "conversations":
        if id_key:
            agent_id, phone = id_key.split(":")
            key = f"convo:{agent_id}:{phone}"
            messages = [json.loads(msg) for msg in r.lrange(key, 0, -1)]
            meta_key = f"convo_meta:{agent_id}:{phone}"
            meta = dict(r.hgetall(meta_key))
            if not messages and not meta:
                return {"messages": [], "meta": {}}
            # Add defaults to meta
            meta.setdefault("agent_id", agent_id)
            meta.setdefault("phone", phone)
            meta["created_at"] = int(meta.get("created_at", 0))
            meta["updated_at"] = int(meta.get("updated_at", 0))
            if filters:
                filtered_messages = [msg for msg in messages if filter_record(msg, filters)]
                if not filtered_messages:
                    return {"messages": [], "meta": meta}
                return {"messages": filtered_messages, "meta": meta}
            return {"messages": messages, "meta": meta}
        
        all_keys = r.smembers("conversations")
        convos = {}
        for ck in all_keys:
            agent_id, phone = ck.split(":")
            key = f"convo:{agent_id}:{phone}"
            messages = [json.loads(msg) for msg in r.lrange(key, 0, -1)]
            meta_key = f"convo_meta:{agent_id}:{phone}"
            meta = dict(r.hgetall(meta_key))
            if messages or meta:
                meta.setdefault("agent_id", agent_id)
                meta.setdefault("phone", phone)
                meta["created_at"] = int(meta.get("created_at", 0))
                meta["updated_at"] = int(meta.get("updated_at", 0))
                if not filters or any(filter_record(msg, filters) for msg in messages):
                    convos[ck] = {"messages": messages, "meta": meta}
        return convos if convos else {}
    
    return {"error": f"Invalid table: {table}"}

if __name__ == "__main__":
    if len(sys.argv) not in [2, 3, 4]:
        print("Usage: python fetch_record.py <table> [id_key] [filters]")
        print("Examples:")
        print("  python fetch_record.py users  # All users")
        print("  python fetch_record.py users 1  # Specific user")
        print("  python fetch_record.py users name=John  # Filtered users")
        print("  python fetch_record.py agents user_id=1  # Filtered agents")
        sys.exit(1)
    
    table = sys.argv[1]
    id_key = sys.argv[2] if len(sys.argv) > 2 else None
    filter_str = sys.argv[3] if len(sys.argv) == 4 else None
    
    try:
        result = fetch_record(table, id_key, filter_str)
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
