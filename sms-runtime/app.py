import os
from fastapi import FastAPI, Request
import redis

app = FastAPI()

# Redis connection (local or Render)
redis_url = os.getenv('REDIS_URL', 'redis://red-d44f17jipnbc73dqs2k0:6379')
r = redis.from_url(redis_url, decode_responses=True)

# Your routes (e.g., /runtime, /users) here...
@app.get("/")
def root():
    return {"message": "SMS Runtime Backend Live on Render!"}

# Example /users fetch
@app.get("/users")
def fetch_users():
    all_ids = r.smembers("users")
    users = {uid: dict(r.hgetall(f"user:{uid}")) for uid in all_ids}
    return users

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
