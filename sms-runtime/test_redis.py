import redis
import json
import time

# Connect to local Docker Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Test Agent Fetch (Hash)
r.hset('agent:test1', mapping={'prompt': 'SMS bot for leads.', 'tools': json.dumps(['cal.com'])})
agent_start = time.time()
agent = r.hgetall('agent:test1')
agent['tools'] = json.loads(agent['tools'])
agent_latency = (time.time() - agent_start) * 1000
print(f"Agent Fetch: {agent} | Latency: {agent_latency}ms")

# Test Convo Fetch/Update (List)
key = 'convo:test1:+1234567890'
r.rpush(key, json.dumps({'role': 'user', 'content': 'Hi'}))
convo_start = time.time()
history = [json.loads(msg) for msg in r.lrange(key, 0, -1)]
convo_latency = (time.time() - convo_start) * 1000
print(f"Convo Fetch: {history} | Latency: {convo_latency}ms")

# Update Convo
update_start = time.time()
r.rpush(key, json.dumps({'role': 'assistant', 'content': 'Hello back!'}))
r.expire(key, 86400)  # 24h TTL
update_latency = (time.time() - update_start) * 1000
print(f"Convo Update Latency: {update_latency}ms")