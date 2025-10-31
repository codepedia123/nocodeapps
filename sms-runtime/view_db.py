#!/usr/bin/env python3
"""
View Redis Database as Tables
- Lists all tables (sets).
- For each, shows records with fields (hashes/lists).
- Formatted for readability.
"""

import redis
import json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def view_db():
    print("=== Redis Database View (Tables, Fields, Records) ===\n")
    
    # Tables (Sets)
    tables = {
        "users": r.smembers("users"),
        "agents": r.smembers("agents"),
        "conversations": r.smembers("conversations")
    }
    
    for table, ids in tables.items():
        print(f"📁 Table: {table.upper()}")
        if not ids:
            print("  (Empty)")
            continue
        
        for id_key in ids:
            if table == "users":
                record = r.hgetall(f"user:{id_key}")
                record["last_active"] = int(record.get("last_active", 0))
                print(f"  📄 Record ID: {id_key}")
                print(f"    Fields: {json.dumps(record, indent=4)}")
            
            elif table == "agents":
                record = r.hgetall(id_key)
                record["tools"] = json.loads(record.get("tools", "[]"))
                print(f"  📄 Record ID: {id_key}")
                print(f"    Fields: {json.dumps(record, indent=4)}")
            
            elif table == "conversations":
                agent_id, phone = id_key.split(":")
                key = f"convo:{agent_id}:{phone}"
                messages = [json.loads(msg) for msg in r.lrange(key, 0, -1)]
                print(f"  📄 Record ID: {id_key}")
                print(f"    Messages: {json.dumps(messages, indent=4)}")
        
        print("\n" + "="*60 + "\n")
    
    print(f"Total Keys: {r.dbsize()}")
    print("=== View Complete! ===")

if __name__ == "__main__":
    try:
        view_db()
    except redis.exceptions.ConnectionError as e:
        print(f"Redis Connection Error: {e}")
        print("Start container: docker start sms-redis")
    except Exception as e:
        print(f"Error: {e}")