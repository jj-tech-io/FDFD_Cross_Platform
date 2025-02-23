import runpod
import json
import os

# Set your RunPod API key directly.
runpod.api_key = "rpa_LZDP7XDD2ETN850HZ57KGIKDQRMITGWKW202YTHQ1jq8uy"

# Use your provided RunPod VS Code Server Endpoint ID.
endpoint_id = "m72g67gm1ygp67"

# Create an Endpoint instance using your endpoint ID.
endpoint = runpod.Endpoint(endpoint_id)

# Prepare a sample payload for the job.
payload = {
    "prompt": "Hello from RunPod VS Code Server with 4 × RTX 4090!"
}

try:
    # Run the job synchronously (waiting up to 60 seconds for completion)
    output = endpoint.run_sync(payload, timeout=60)
    print("Job Output:")
    print(json.dumps(output, indent=2))
except Exception as e:
    print("An error occurred while running the job:")
    print(e)
