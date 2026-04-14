"""
MBPP+ evaluation via OpenRouter API.
3-shot prompting with code extraction and sandboxed execution.
pass@1: 1 if generated code passes all EvalPlus adversarial tests, 0 otherwise.
"""

import json
import re
import random
from datasets import load_dataset
from open_router_tasks.common import BaseOpenRouterTask
from open_router_tasks.mbppplus_prompt import FEW_SHOT_EXAMPLES
from nanollama.execution import execute_code

EXEC_TIMEOUT = 60.0  # seconds; worst-case reference solution ~37s

# --- Code extraction ---

_PYTHON_BLOCK_RE = re.compile(r"```python\s*\n(.*?)(?:\n```|$)", re.DOTALL)
_ANY_BLOCK_RE = re.compile(r"```(?:\w+)?\s*\n(.*?)(?:\n```|$)", re.DOTALL)


def extract_code(completion):
    """Extract Python code from model completion.
    Priority: ```python block > any ``` block > raw completion."""
    m = _PYTHON_BLOCK_RE.search(completion)
    if m:
        return m.group(1).strip()
    m = _ANY_BLOCK_RE.search(completion)
    if m:
        return m.group(1).strip()
    stripped = completion.strip()
    return stripped if stripped else None


def _compose_code(test_imports, generated_code, test_harness):
    """Compose executable code: imports + generated function + test harness."""
    parts = []
    if test_imports:
        parts.append("\n".join(test_imports))
    parts.append(generated_code)
    parts.append(test_harness)
    return "\n\n".join(parts)


# --- Task class ---

class MBPPPlusOpenRouter(BaseOpenRouterTask):

    def __init__(self, model, max_tokens=2**16, temperature=0.0, reasoning=False, provider=None, log_dir=None):
        super().__init__(model, max_tokens=max_tokens, temperature=temperature,
                         reasoning=reasoning, provider=provider, log_dir=log_dir)
        self.test_ds = load_dataset("evalplus/mbppplus", split="test")

    def _log_filename(self):
        return f"{self.model.replace('/', '_')}_mbppplus.jsonl"

    def build_messages(self, row):
        messages = []
        for user_content, assistant_content in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": assistant_content})

        tests_str = "\n".join(row["test_list"])
        final_user = (
            f"You are an expert Python programmer, and here is your task:\n"
            f"{row['prompt']}\n"
            f"Your code should pass the following tests:\n{tests_str}"
        )
        messages.append({"role": "user", "content": final_user})
        return messages

    def _eval_single(self, i):
        row = self.test_ds[i]
        messages = self.build_messages(row)
        completion = ""
        try:
            response = self._chat(messages)
            completion = response["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"\nError on problem {i} (task_id={row['task_id']}): {e}")

        generated_code = extract_code(completion)
        is_correct = False
        exec_result = None

        if generated_code:
            code_to_exec = _compose_code(row["test_imports"], generated_code, row["test"])
            exec_result = execute_code(code_to_exec, timeout=EXEC_TIMEOUT)
            is_correct = exec_result.success

        self._log_result({
            "index": i,
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "generated_code": generated_code,
            "correct": is_correct,
            "exec_success": exec_result.success if exec_result else False,
            "exec_error": exec_result.error if exec_result else "no_code_extracted",
            "exec_timeout": exec_result.timeout if exec_result else False,
            "raw_response": completion,
        })
        return i, is_correct

    def debug_single(self, seed=None):
        """Run on one random problem, printing payload, response, code, execution, and grading."""
        SEP = "=" * 80
        i = random.Random(seed).randrange(len(self.test_ds))
        row = self.test_ds[i]
        messages = self.build_messages(row)

        print(SEP)
        print(f"DEBUG: MBPP_PLUS  |  Problem #{i} of {len(self.test_ds)}  "
              f"[task_id={row['task_id']}]")
        print(SEP)
        print(f"\n[TASK]\n{row['prompt']}")
        print(f"\n[TEST LIST (shown in prompt)]")
        for t in row["test_list"]:
            print(f"  {t}")
        print(f"\n[REFERENCE CODE]\n{row['code']}")
        print(f"\n[PAYLOAD -- {len(messages)} messages]")
        print(json.dumps(messages, indent=2))

        response = self._chat(messages)
        print(f"\n[API RESPONSE]")
        print(json.dumps(response, indent=2))

        completion = response["choices"][0]["message"]["content"]
        generated_code = extract_code(completion)

        print(f"\n[COMPLETION]\n{completion}")
        print(f"\n[EXTRACTED CODE]\n{generated_code}")

        if generated_code:
            code_to_exec = _compose_code(row["test_imports"], generated_code, row["test"])
            print(f"\n[COMPOSED EXECUTION CODE]\n{code_to_exec}")
            print(f"\n[EXECUTING with timeout={EXEC_TIMEOUT}s ...]")
            exec_result = execute_code(code_to_exec, timeout=EXEC_TIMEOUT)
            is_correct = exec_result.success
            print(f"\n[EXECUTION RESULT]\n{exec_result}")
        else:
            is_correct = False
            print("\n[EXECUTION RESULT]  SKIPPED -- no code extracted")

        print(f"\n[GRADING]  {'PASS (pass@1=1)' if is_correct else 'FAIL (pass@1=0)'}")
        print(SEP)
