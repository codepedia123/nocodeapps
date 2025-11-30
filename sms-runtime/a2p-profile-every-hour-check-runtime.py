# a2p_review_worker.py

import requests
import time
import threading
import traceback
import json
from datetime import datetime

# =====================================================================
# INJECTION FROM /int
# =====================================================================
# /int injects variables into globals()
inputs = globals()
action = inputs.get("action")
payload = inputs.get("payload", {})
# Some workflows may send everything in "input"
input_data = inputs.get("input", {})

# Unified working input for main()
combined_input = {}

# priority: payload > input_data > raw globals
if isinstance(payload, dict):
    combined_input.update(payload)
if isinstance(input_data, dict):
    combined_input.update(input_data)


# =====================================================================
# ORIGINAL SYSTEM CONSTANTS (UNCHANGED)
# =====================================================================
CONNECT_TIMEOUT = 10
READ_TIMEOUT_FETCH = 300
READ_TIMEOUT_DEFAULT = 40
MAX_FETCH_RETRIES = 3
RETRY_SLEEP_SECONDS = 3

BACKEND_BASE_URL = "http://3.90.183.107:8000"
FETCH_TABLE_NAME = "a2p-brands-in-review"
LOG_TABLE = "log"

TRUSTHUB_CUSTOMER_PROFILE_BASE = "https://trusthub.twilio.com/v1/CustomerProfiles"

TRUSTHUB_AUTH_HEADERS = {
    "Authorization": (
        "Basic "
        "QUMzNDYwZTIxYjhhNjg2N2Q4ZmYxZWFmZWFlZDgzZjgzOToyNDcwOWMzOWMy"
        "OTkzNzJmOWY5MDY5MWM5OGUxY2Y4Nw=="
    )
}

FUNCTION_ID_AFTER_APPROVAL = "func1764069347"



# =====================================================================
# LOGGING (UNCHANGED)
# =====================================================================


def log(step, payload=None):
    try:
        if payload is not None and not isinstance(payload, str):
            payload = json.dumps(payload)  # FIX HERE

        body = {
            "table": LOG_TABLE,
            "data": {
                "step": step,
                "payload": payload or "{}",
                "timestamp": datetime.utcnow().isoformat(),
            }
        }

        requests.post(
            BACKEND_BASE_URL + "/add",
            json=body,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT_DEFAULT)
        )
    except Exception:
        pass




# =====================================================================
# SAFE JSON PARSER (UNCHANGED)
# =====================================================================
def safe_get_json(response, context):
    try:
        return response.json()
    except Exception:
        try:
            status = response.status_code
        except Exception:
            status = "unknown"
        raise Exception(f"{context}: invalid JSON response (status {status})")



# =====================================================================
# FETCH RECORDS (UNCHANGED)
# =====================================================================
def fetch_in_review_records():
    log("fetch_start", {})
    url = BACKEND_BASE_URL + "/fetch"
    params = {"table": FETCH_TABLE_NAME}

    attempt = 0
    while attempt < MAX_FETCH_RETRIES:
        attempt += 1
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT_FETCH),
            )
            log("fetch_response", {"status": resp.status_code, "text": resp.text})
        except requests.exceptions.Timeout:
            log("fetch_timeout", {"attempt": attempt})
            if attempt < MAX_FETCH_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)
                continue
            raise Exception(
                "Fetch in-review records timed-out after multiple attempts "
                f"(max {READ_TIMEOUT_FETCH}s read window)"
            )
        except Exception as exc:
            raise Exception(f"Fetch in-review records request failed: {str(exc)}")

        if not resp.ok:
            raise Exception(
                f"Fetch in-review records failed with status "
                f"{resp.status_code}: {resp.text}"
            )

        data = safe_get_json(resp, "Fetch in-review records")

        if data is None:
            return {}

        try:
            _ = data.keys
            _ = data.get
        except Exception:
            raise Exception("Fetch in-review records returned non-dict payload")

        try:
            candidate = data.get("data")
            _ = candidate.keys
            _ = candidate.get
            records = candidate
        except Exception:
            records = data

        try:
            _ = records.keys
            _ = records.get
        except Exception:
            raise Exception("In-review records payload is not a dict")

        return records

    raise Exception("Unexpected fetch retry loop exit")



# =====================================================================
# TWILIO CUSTOMER PROFILE STATUS (UNCHANGED)
# =====================================================================
def fetch_customer_profile_status(customer_profile_sid):
    if not customer_profile_sid:
        raise Exception("Customer profile SID is missing")

    url = f"{TRUSTHUB_CUSTOMER_PROFILE_BASE}/{customer_profile_sid}"
    log("twilio_request_sent", {"url": url})

    try:
        resp = requests.get(
            url,
            headers=TRUSTHUB_AUTH_HEADERS,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT_DEFAULT),
        )
        log("twilio_response_received", {"status": resp.status_code, "text": resp.text})
    except Exception as exc:
        raise Exception(
            f"Fetch Customer Profile {customer_profile_sid} request failed: {str(exc)}"
        )

    data = safe_get_json(resp, f"Fetch Customer Profile {customer_profile_sid}")

    if not resp.ok:
        raise Exception(
            f"Fetch Customer Profile {customer_profile_sid} failed: {data}"
        )

    return data.get("status")



# =====================================================================
# DELETE APPROVED RECORD (UNCHANGED)
# =====================================================================
def delete_review_record(record_id):
    url = BACKEND_BASE_URL + "/delete"
    params = {"table": FETCH_TABLE_NAME, "id": record_id}

    log("delete_call_sent", {"record_id": record_id})

    try:
        resp = requests.delete(
            url, params=params, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT_DEFAULT)
        )
        log("delete_call_response", {"status": resp.status_code, "text": resp.text})
    except Exception as exc:
        raise Exception(f"Delete review record {record_id} request failed: {str(exc)}")

    if not resp.ok:
        raise Exception(
            f"Delete review record {record_id} failed "
            f"(status {resp.status_code}): {resp.text}"
        )



# =====================================================================
# FOLLOW-UP FUNCTION CALL (UNCHANGED)
# =====================================================================
def trigger_post_approval_function(record):
    url = BACKEND_BASE_URL + "/runfunction"

    wanted = [
        "customer_profile_sid",
        "enduser_sid",
        "address_sid",
        "supporting_document_sid",
        "friendly_name",
        "email",
        "first_name",
        "last_name",
        "phone_number",
        "street",
        "street_secondary",
        "city",
        "region",
        "postal_code",
        "country",
        "policy_sid",
        "user",
    ]

    input_payload = {}
    for key in wanted:
        val = record.get(key)
        if val is not None:
            input_payload[key] = val

    if "user" not in input_payload:
        raise Exception("Record missing 'user' field; cannot trigger follow-up")

    body = {"id": FUNCTION_ID_AFTER_APPROVAL, "input": input_payload}

    log("trigger_function_start", body)

    try:
        resp = requests.post(
            url, json=body, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT_DEFAULT)
        )
        log("trigger_function_response", {"status": resp.status_code, "text": resp.text})
    except Exception as exc:
        raise Exception(f"runfunction request failed: {str(exc)}")

    if not resp.ok:
        raise Exception(
            f"runfunction call failed (status {resp.status_code}): {resp.text}"
        )

    return safe_get_json(resp, "runfunction response")



# =====================================================================
# MAIN ORCHESTRATION (UNCHANGED)
# =====================================================================
def main(_):
    log("job_main_start", {})

    records = fetch_in_review_records()
    if not records:
        log("job_no_records", {})
        return {"status": "no-records", "message": "No rows in review table"}

    sorted_ids = sorted(
        records.keys(),
        key=lambda k: (0, int(k)) if str(k).isdigit() else (1, str(k))
    )

    last_json = None
    approved_found = False

    for rec_id in sorted_ids:
        rec = records.get(rec_id)
        try:
            _ = rec.keys
            _ = rec.get
        except Exception:
            continue

        sid = rec.get("customer_profile_sid")
        if not sid:
            continue

        status = fetch_customer_profile_status(sid)

        if status == "twilio-approved":
            approved_found = True
            delete_review_record(rec_id)
            last_json = trigger_post_approval_function(rec)

    log("job_completed", {"approved_found": approved_found})

    if approved_found and last_json is not None:
        return last_json

    return {
        "status": "no-approved-profiles",
        "message": "Checked but none approved",
        "checked_record_ids": sorted_ids,
    }



# =====================================================================
# BACKGROUND EXECUTION GLUE  (UPDATED BUT LOGIC SAME)
# =====================================================================
def background_runner(inputs):
    try:
        result = main(inputs)
        log("background_final_result", {"result": result})
    except Exception as e:
        log("background_error", {
            "error": str(e),
            "trace": traceback.format_exc()
        })


# Always run background
threading.Thread(
    target=background_runner,
    args=(combined_input,),
    daemon=True
).start()


# =====================================================================
# IMMEDIATE RETURN (UPDATED)
# =====================================================================
result = {
    "status": "started",
    "job_id": f"job_{int(time.time())}"
}
