#!/usr/bin/env python3
import redis
import json
import sys
import time
from typing import Dict, Any

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def update_record(table: str, id_key: str, updates: Dict[str, Any]) -> bool:
    updates["updated_at"] = int(time.time())

    if table == "users":
        key = f"user:{id_key}"
        if not r.exists(key):
            print(f"User {id_key} not found")
            return False
        r.hset(key, mapping=updates)
        print(f"✅ Updated user {id_key}")
        return True

    elif table == "agents":
        key = id_key  # e.g., "agent1"
        if not r.exists(key):
            print(f"Agent {id_key} not found")
            return False
        if "tools" in updates:
            updates["tools"] = json.dumps(updates["tools"])
        r.hset(key, mapping=updates)
        print(f"✅ Updated agent {id_key}")
        return True

    elif table == "conversations":
        agent_id, phone = id_key.split(":", 1)
        key = f"convo:{agent_id}:{phone}"
        meta_key = f"convo_meta:{agent_id}:{phone}"
        if not r.exists(key):
            print(f"Conversation {id_key} not found")
            return False
        if "append" in updates:
            for msg in updates["append"]:
                r.rpush(key, json.dumps(msg))
            r.expire(key, 86400 * 30)
            r.hset(meta_key, "updated_at", updates["updated_at"])
            print(f"✅ Appended {len(updates['append'])} messages to {id_key}")
            return True
        else:
            r.hset(meta_key, mapping=updates)
            print(f"✅ Updated conversation metadata {id_key}")
            return True

    print(f"Invalid table: {table}")
    return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python update_record.py <table:id_key> '<json_updates>'")
        sys.exit(1)

    try:
        table_id = sys.argv[1]
        updates_str = sys.argv[2]
        table, id_key = table_id.split(":", 1)
        updates = json.loads(updates_str)
        success = update_record(table, id_key, updates)
        print("Success" if success else "Failed")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)