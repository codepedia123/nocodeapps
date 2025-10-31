#!/usr/bin/env python3
import redis
import json
import sys
import time
from typing import Dict, Any

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def add_record(table: str, data: Dict[str, Any]) -> str:
    data["created_at"] = int(time.time())
    data["updated_at"] = data["created_at"]

    if table == "users":
        next_id = int(r.get("next_user_id") or 0) + 1
        r.set("next_user_id", next_id)
        user_id = str(next_id)
        key = f"user:{user_id}"
        r.hset(key, mapping=data)
        r.sadd("users", user_id)
        print(f"✅ Added user {user_id}")
        return user_id

    elif table == "agents":
        next_id = int(r.get("next_agent_id") or 0) + 1
        r.set("next_agent_id", next_id)
        agent_id = f"agent{next_id}"
        if "tools" in data:
            data["tools"] = json.dumps(data["tools"])
        r.hset(agent_id, mapping=data)
        r.sadd("agents", agent_id)
        print(f"✅ Added agent {agent_id}")
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
        print(f"✅ Added conversation {agent_id}:{phone}")
        return f"{agent_id}:{phone}"

    else:
        raise ValueError(f"Invalid table: {table}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python add_record.py <table> '<json_data>'")
        sys.exit(1)

    table = sys.argv[1]
    try:
        data = json.loads(sys.argv[2])
        result = add_record(table, data)
        print(f"Success: {result}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)