#!/bin/bash

# Navigate to SMS runtime folder
cd /Users/suyashsah/sms-runtime || { echo "Folder not found"; exit 1; }

# Check and create/recreate venv with --copies
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi
echo "Creating new virtual environment..."
python3 -m venv .venv --copies
source .venv/bin/activate

# Install redis if not present
if ! python -c "import redis" 2>/dev/null; then
    echo "Installing redis==5.0.1..."
    python -m pip install redis==5.0.1
    echo "redis==5.0.1" > requirements.txt
    python -m pip install -r requirements.txt
else
    echo "redis already installed."
fi

# Upgrade pip
python -m pip install --upgrade pip

# Start or recreate Redis container
if ! docker ps -q -f name=sms-redis | grep -q .; then
    echo "Recreating sms-redis container..."
    docker rm -f sms-redis 2>/dev/null
    docker run --name sms-redis -p 6379:6379 -d redis:8-alpine redis-server --appendonly yes
else
    echo "Starting existing sms-redis container..."
    docker start sms-redis
fi

# Wait for Redis to be ready
echo "Waiting for Redis to start..."
until redis-cli -h localhost -p 6379 ping | grep -q PONG; do
    sleep 1
done
echo "Redis is ready!"

# Run redisset.py to reset and initialize
echo "Initializing Redis database..."
./.venv/bin/python redisset.py

# Verify with view_db.py
echo "Verifying database setup..."
if [ ! -f "view_db.py" ]; then
    echo "Creating view_db.py..."
    cat > view_db.py << 'EOF'
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
EOF
    chmod +x view_db.py
fi
./.venv/bin/python view_db.py

echo "=== SMS Runtime Setup Complete! ==="
echo "Activate venv manually with: source .venv/bin/activate"
echo "Run scripts with: ./.venv/bin/python <script>.py"