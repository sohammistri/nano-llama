import os
import json
import time
import requests

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')

def chat(messages, model, max_tokens=512, temperature=0.0, reasoning=False, retries=3):
    """Call OpenRouter chat completions API with retry on rate limit."""
    assert OPENROUTER_API_KEY, "Set OPENROUTER_API_KEY environment variable"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if reasoning:
        payload["reasoning"] = {"enabled": True}
    for attempt in range(retries):
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload)
        )
        if response.status_code != 429:
            break
        wait = 2 ** attempt
        print(f"Rate limited, retrying in {wait}s...")
        time.sleep(wait)
    response.raise_for_status()
    return response.json()
