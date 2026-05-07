import requests
import json
import os

import time

OPENROUTER_API_KEY = os.environ['OPENROUTER_API_KEY']

def chat(messages, model="google/gemma-4-31b-it", reasoning=True, retries=3):
  payload = {"model": model, "messages": messages}
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

# LLaMA 3.2 3B - non-reasoning model
LLAMA_MODEL = "meta-llama/llama-3.2-3b-instruct"

SYSTEM_PROMPT = "You are an expert at solving high school level math problems."
INSTRUCTION = "Let's think over the problem step by step. At the end, you MUST write the answer as an integer after '#### '."

question = "A candle melts by 2 centimeters every hour that it burns. How many centimeters shorter will a candle be after burning from 1:00 PM to 5:00 PM?"

llama_response = chat(
  messages=[
    {"role": "system", "content": f"{SYSTEM_PROMPT}"},
    {"role": "user", "content": f"{question}\n\n{INSTRUCTION}"}
  ],
  model=LLAMA_MODEL,
  reasoning=False
)
llama_message = llama_response['choices'][0]['message']
print(f"LLaMA Response 1: {json.dumps(llama_message, indent=2)}")

# llama_messages = [
#   {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
#   {"role": "assistant", "content": llama_message.get('content')},
#   {"role": "user", "content": "Are you sure? Think carefully."}
# ]

# llama_response2 = chat(messages=llama_messages, model=LLAMA_MODEL, reasoning=False)
# print(f"LLaMA Response 2: {json.dumps(llama_response2, indent=2)}")