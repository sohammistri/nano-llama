"""
GSM8K evaluation via OpenRouter API.
Supports n-shot prompting with few-shot examples from the train split.
"""

import os
import json
import re
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from open_router_tasks.common import chat

SYSTEM_PROMPT = "You are an expert at solving high school level math problems."
INSTRUCTION = "Let's think over the problem step by step. At the end, you MUST write the answer as an integer after '#### '."

GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_answer(text):
    match = GSM_RE.search(text)
    if match:
        return match.group(1).strip().replace(",", "")
    return None


class GSM8KOpenRouter:

    def __init__(self, model, n_shot=8, max_tokens=2**16, temperature=0.0, reasoning=False, log_dir=None):
        self.model = model
        self.n_shot = n_shot
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning = reasoning
        self.log_dir = log_dir
        self._log_lock = threading.Lock()
        self._log_file = None

        self.test_ds = load_dataset("openai/gsm8k", "main", split="test")
        if n_shot > 0:
            train_ds = load_dataset("openai/gsm8k", "main", split="train")
            indices = random.Random(42).sample(range(len(train_ds)), n_shot)
            self.few_shot_examples = [(train_ds[i]['question'], train_ds[i]['answer']) for i in indices]
        else:
            self.few_shot_examples = []

    def build_messages(self, question):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for fs_question, fs_answer in self.few_shot_examples:
            messages.append({"role": "user", "content": fs_question})
            messages.append({"role": "assistant", "content": fs_answer})
        messages.append({"role": "user", "content": f"{question}\n\n{INSTRUCTION}"})
        return messages

    def _log_result(self, record):
        if self._log_file is None:
            return
        with self._log_lock:
            self._log_file.write(json.dumps(record) + "\n")
            self._log_file.flush()

    def _eval_single(self, i):
        row = self.test_ds[i]
        question = row['question']
        ref_answer = extract_answer(row['answer'])
        messages = self.build_messages(question)
        try:
            response = chat(
                messages, model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                reasoning=self.reasoning,
            )
            completion = response['choices'][0]['message']['content']
        except Exception as e:
            print(f"\nError on problem {i}: {e}")
            completion = ""
        pred_answer = extract_answer(completion)
        is_correct = (pred_answer is not None) and (pred_answer == ref_answer)
        self._log_result({
            "index": i,
            "question": question,
            "ground_truth": ref_answer,
            "predicted": pred_answer,
            "correct": is_correct,
            "raw_response": completion,
        })
        return i, is_correct

    def debug_single(self, seed=None):
        """Run on one random problem, printing payload, response, and grading."""
        SEP = "=" * 80
        i = random.Random(seed).randrange(len(self.test_ds))
        row = self.test_ds[i]
        question = row['question']
        ref_answer = extract_answer(row['answer'])
        messages = self.build_messages(question)

        print(SEP)
        print(f"DEBUG: GSM8K  |  Problem #{i} of {len(self.test_ds)}")
        print(SEP)
        print(f"\n[QUESTION]\n{question}")
        print(f"\n[GROUND TRUTH]\n{ref_answer}")
        print(f"\n[PAYLOAD — {len(messages)} messages]")
        print(json.dumps(messages, indent=2))

        response = chat(messages, model=self.model, max_tokens=self.max_tokens,
                        temperature=self.temperature, reasoning=self.reasoning)
        print(f"\n[API RESPONSE]")
        print(json.dumps(response, indent=2))

        completion = response['choices'][0]['message']['content']
        pred_answer = extract_answer(completion)
        is_correct = (pred_answer is not None) and (pred_answer == ref_answer)

        print(f"\n[COMPLETION]\n{completion}")
        print(f"\n[EXTRACTED ANSWER]  {pred_answer}")
        print(f"\n[GRADING]  {'CORRECT' if is_correct else 'INCORRECT'}  "
              f"(pred={pred_answer}, ref={ref_answer})")
        print(SEP)

    def run_eval(self, max_problems=None, workers=10):
        num_problems = len(self.test_ds) if max_problems is None else min(len(self.test_ds), max_problems)
        num_correct, total = 0, 0

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            log_path = os.path.join(self.log_dir, f"{self.model.replace('/', '_')}_n{self.n_shot}.jsonl")
            self._log_file = open(log_path, "w")
            print(f"Logging to: {log_path}")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._eval_single, i): i for i in range(num_problems)}
            for future in as_completed(futures):
                _, is_correct = future.result()
                total += 1
                num_correct += int(is_correct)
                print(f"\r\033[K{num_correct}/{total} ({100*num_correct/total:.2f}%)", end='', flush=True)

        if self._log_file:
            self._log_file.close()
            self._log_file = None

        print()
        print("=" * 50)
        accuracy = num_correct / total if total > 0 else 0.0
        print(f"maj@1: {num_correct}/{total} ({100*accuracy:.2f}%)")
        return accuracy
