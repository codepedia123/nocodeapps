import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import redis
import json
from typing import Dict, Any
from pydantic import BaseModel

# Redis connection (Render env or local fallback)
redis_url = os.getenv('REDIS_URL', 'redis://red-d44f17jipnbc73dqs2k0:6379')
r = redis.from_url(redis_url, decode_responses=True)

app = FastAPI(title="SMS Runtime Backend")

# CORS for frontend (optional)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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

# --- ADD ROUTE ---
@app.post("/add")
def add_endpoint(request: AddRequest):
    try:
        result_id = add_record(request.table, request.data)
        return {"id": result_id, "status": "success"}
    except Exception as e:
        return {"error": str(e)}, 400

# --- FETCH ROUTE ---
@app.get("/fetch")
def fetch_endpoint(table: str, id_key: Optional[str] = None, filters: Optional[str] = None):
    try:
        result = fetch_record(table, id_key, filters)
        return result
    except Exception as e:
        return {"error": str(e)}, 400

# --- UPDATE ROUTE ---
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

# --- DELETE ROUTE ---
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

# Root health check
@app.get("/")
def root():
    return {"message": "SMS Runtime Backend Live on Render!"}

# Example /users (from earlier)
@app.get("/users")
def get_users():
    try:
        all_ids = r.smembers("users")
        users = {uid: dict(r.hgetall(f"user:{uid}")) for uid in all_ids}
        return users
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
