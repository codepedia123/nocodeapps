#!/usr/bin/env python3
"""
Redis Database Initializer for SMS Agent Runtime
- Resets the database (FLUSHDB).
- Initializes counters and empty sets for Users, Agents, and Conversations.
- Verifies setup with a summary.
- Note: Sets are created lazily by CRUD scripts; no initial SADD with empty lists.
"""

import redis
import sys

# Connect to Redis (local Docker setup)
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def reset_and_initialize():
    print("=== Redis Database Reset & Initialization ===")
    
    # Step 1: Confirmation for Reset
    print("WARNING: This will DELETE ALL existing data (keys, sets, counters).")
    confirmation = input("Type 'RESET' to confirm: ").strip().upper()
    if confirmation != "RESET":
        print("Aborted. No changes made.")
        sys.exit(0)
    
    # Step 2: Flush the Database (Removes All "Tables" and Fields)
    print("Flushing database (FLUSHDB)...")
    r.flushdb()
    print("✅ Database cleared. All keys deleted.")
    
    # Step 3: Create from Starting (Counters Only)
    print("Initializing counters...")
    r.set("next_user_id", "0")
    r.set("next_agent_id", "0")
    print("✅ Counters set to 0.")
    
    # Note: Sets (users, agents, conversations) are created lazily by add_record.py
    # No SADD with empty lists needed—avoids Invalid input error
    
    # Step 4: Verification Summary
    print("\n=== Verification ===")
    print(f"Next User ID: {r.get('next_user_id')}")
    print(f"Next Agent ID: {r.get('next_agent_id')}")
    print(f"Users Set: {r.smembers('users')}")
    print(f"Agents Set: {r.smembers('agents')}")
    print(f"Conversations Set: {r.smembers('conversations')}")
    print(f"Total Keys: {r.dbsize()}")
    print("=== Setup Complete! Ready for CRUD scripts. ===")

if __name__ == "__main__":
    try:
        reset_and_initialize()
    except redis.exceptions.ConnectionError as e:
        print(f"Redis Connection Error: {e}")
        print("Ensure Docker Redis is running: docker start sms-redis")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected Error: {e}")
        sys.exit(1)