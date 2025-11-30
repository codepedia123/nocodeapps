# process_lead.py

# This script demonstrates how a standalone job file should be written
# so that the /int?file=process_lead endpoint can run it cleanly.

# These variables may be injected later or replaced by environment-style config.
action = globals().get("action", None)
payload = globals().get("payload", {})

def log(msg):
    print("[process_lead]", msg)

def run_action():
    if action is None:
        log("No action provided to this script")
        return

    if action == "validate":
        name = payload.get("name")
        phone = payload.get("phone")

        if not name or not phone:
            log("Validation failed: Missing fields")
            return

        if len(phone) != 10:
            log("Validation failed: Invalid phone number")
            return

        log(f"Lead validated: {name} ({phone})")
        return

    if action == "score":
        lead_score = 0
        
        if payload.get("income"):
            lead_score += 20
        if payload.get("age") and payload["age"] > 25:
            lead_score += 30
        if payload.get("state") in ["NY", "CA", "TX"]:
            lead_score += 40

        log(f"Lead score calculated: {lead_score}")
        return

    log(f"Unknown action: {action}")

# Run the handler
if __name__ == "__main__":
    run_action()
