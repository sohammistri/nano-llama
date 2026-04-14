import os
import json
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')

def chat(messages, model, max_tokens=2**16, temperature=0.0, reasoning=False, retries=3):
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
    if not response.ok:
        raise requests.HTTPError(
            f"HTTP {response.status_code}: {response.text}",
            response=response,
        )
    return response.json()


class BaseOpenRouterTask:
    """Base class for OpenRouter chat evaluation tasks.

    Subclasses must:
      - Set self.test_ds in __init__ (used by run_eval for len/indexing)
      - Implement _eval_single(i) -> (i, is_correct)
      - Implement debug_single(seed=None)
      - Implement build_messages(row)
      - Override _log_filename() if the default slug is insufficient
    """

    # After this many consecutive 400s at the current max_tokens, halve it.
    _MAX_TOKENS_400_THRESHOLD = 10

    def __init__(self, model, max_tokens=2**16, temperature=0.0, reasoning=False, log_dir=None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning = reasoning
        self.log_dir = log_dir
        self._log_lock = threading.Lock()
        self._log_file = None
        self._token_lock = threading.Lock()
        self._consecutive_400s = 0

    def _chat(self, messages):
        """Call the API, halving max_tokens after >10 consecutive 400s."""
        while True:
            current_max_tokens = self.max_tokens
            try:
                result = chat(
                    messages,
                    model=self.model,
                    max_tokens=current_max_tokens,
                    temperature=self.temperature,
                    reasoning=self.reasoning,
                )
                with self._token_lock:
                    self._consecutive_400s = 0
                return result
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 400:
                    with self._token_lock:
                        self._consecutive_400s += 1
                        should_halve = self._consecutive_400s > self._MAX_TOKENS_400_THRESHOLD
                        if should_halve:
                            self.max_tokens = max(self.max_tokens // 2, 256)
                            self._consecutive_400s = 0
                    if should_halve:
                        print(f"\nHalved max_tokens to {self.max_tokens} after "
                              f"{self._MAX_TOKENS_400_THRESHOLD} consecutive 400s, retrying...")
                        continue
                raise

    def _log_filename(self):
        """JSONL filename (no directory). Override in subclasses for task-specific suffixes."""
        return f"{self.model.replace('/', '_')}.jsonl"

    def _log_result(self, record):
        if self._log_file is None:
            return
        with self._log_lock:
            self._log_file.write(json.dumps(record) + "\n")
            self._log_file.flush()

    def _eval_single(self, i):
        raise NotImplementedError

    def debug_single(self, seed=None):
        raise NotImplementedError

    def run_eval(self, max_problems=None, workers=10):
        num_problems = len(self.test_ds) if max_problems is None else min(len(self.test_ds), max_problems)
        num_correct, total = 0, 0

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            log_path = os.path.join(self.log_dir, self._log_filename())
            self._log_file = open(log_path, "w")
            print(f"Logging to: {log_path}")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._eval_single, i): i for i in range(num_problems)}
            for future in as_completed(futures):
                _, is_correct = future.result()
                total += 1
                num_correct += int(is_correct)
                print(f"\r\033[K{num_correct}/{total} ({100*num_correct/total:.2f}%)", end="", flush=True)

        if self._log_file:
            self._log_file.close()
            self._log_file = None

        print()
        print("=" * 50)
        accuracy = num_correct / total if total > 0 else 0.0
        print(f"Accuracy: {num_correct}/{total} ({100*accuracy:.2f}%)")
        return accuracy
