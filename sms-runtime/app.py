import os
from typing import Optional, Dict, Any, Union
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

# Import CRUD functions (from db_ops.py - assume it's in same folder)
from db_ops import add_record, fetch_record, update_record

# Health check (root)
@app.get("/")
def root():
    try:
        if r and r.ping():
            return {"message": "SMS Runtime Backend Live on Render!", "redis": "Connected"}
        else:
            return {"message": "Backend Live, but Redis down - check connection."}, 503
    except Exception as e:
        return {"error": str(e)}, 500

# Add record
@app.post("/add")
def add_endpoint(request: AddRequest):
    try:
        if not r:
            raise ValueError("Redis not connected")
        result_id = add_record(request.table, request.data)
        return {"id": result_id, "status": "success"}
    except Exception as e:
        return {"error": str(e)}, 400

# Fetch record
@app.get("/fetch")
def fetch_endpoint(table: str, id_key: Optional[str] = None, filters: Optional[str] = None):
    try:
        if not r:
            raise ValueError("Redis not connected")
        result = fetch_record(table, id_key, filters)
        return result
    except Exception as e:
        return {"error": str(e)}, 400

# Update record
@app.post("/update")
def update_endpoint(request: UpdateRequest):
    try:
        if not r:
            raise ValueError("Redis not connected")
        success = update_record(request.table, request.id, request.updates)
        if success:
            return {"status": "success"}
        else:
            return {"error": "Record not found"}, 404
    except Exception as e:
        return {"error": str(e)}, 400

# Delete record
@app.delete("/delete")
def delete_endpoint(table: str, id: str):
    try:
        if not r:
            raise ValueError("Redis not connected")
        success = delete_record(table, id)
        if success:
            return {"status": "deleted"}
        else:
            return {"error": "Record not found"}, 404
    except Exception as e:
        return {"error": str(e)}, 400

# Example /users GET (from earlier)
@app.get("/users")
def get_users():
    try:
        if not r:
            raise ValueError("Redis not connected")
        all_ids = r.smembers("users")
        users = {uid: dict(r.hgetall(f"user:{uid}")) for uid in all_ids}
        return users
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
