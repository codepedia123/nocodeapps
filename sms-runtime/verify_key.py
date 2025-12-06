# verify_key.py
# This file is used ONLY to check exactly what the backend receives.

# "inputs" is injected by /int endpoint from your FastAPI backend.
api_key = inputs.get("api_key")
message = inputs.get("message")
provider = inputs.get("provider")

# Return everything unchanged
result = {
    "received_api_key": api_key,
    "received_length": len(api_key) if isinstance(api_key, str) else None,
    "provider": provider,
    "message": message
}
