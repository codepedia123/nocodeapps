#!/usr/bin/env python3
import redis
import json
import sys
from typing import Dict, Any, Union, Optional

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def parse_filters(filter_str: Optional[str]) -> Dict[str, str]:
    """Parse 'field1=value1,field2=value2' to dict."""
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

def fetch_record(table: str, id_key: Optional[str] = None, filter_str: Optional[str] = None) -> Union[Dict, list]:
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

    else:
        return {"error": f"Invalid table: {table}"}

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage:")
        print("  python fetch_record.py <table>")
        print("  python fetch_record.py <table> <id>")
        print("  python fetch_record.py <table> <field>=<value>[,<field>=<value>...]")
        print("  python fetch_record.py <table> <id> <field>=<value>...")
        print("")
        print("Examples:")
        print("  python fetch_record.py users")
        print("  python fetch_record.py users 1")
        print("  python fetch_record.py users name=Alice")
        print("  python fetch_record.py agents user_id=1")
        print("  python fetch_record.py conversations agent1:+1234567890 role=user")
        sys.exit(1)

    table = sys.argv[1]

    if len(sys.argv) == 2:
        id_key = None
        filter_str = None
    elif len(sys.argv) == 3:
        arg2 = sys.argv[2]
        if '=' in arg2:
            id_key = None
            filter_str = arg2
        else:
            id_key = arg2
            filter_str = None
    else:  # len == 4
        id_key = sys.argv[2]
        filter_str = sys.argv[3]

    try:
        result = fetch_record(table, id_key, filter_str)
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)