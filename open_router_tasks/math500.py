"""
MATH-500 evaluation via OpenRouter API.
Supports n-shot prompting with canonical few-shot examples from Minerva paper.
Answer normalization follows Section D.1 of arXiv 2206.14858.
"""

import os
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from open_router_tasks.common import chat
from open_router_tasks.math500_prompt import FEW_SHOT_EXAMPLES

SYSTEM_PROMPT = "You are an expert at solving competition-level math problems."
INSTRUCTION = "Let's think over the problem step by step. At the end, write your answer in the format: Final Answer: The final answer is $ANSWER$. I hope it is correct."

# --- Answer extraction ---

FINAL_ANSWER_RE = re.compile(
    r"[Ff]inal\s+[Aa]nswer.*?[Tt]he\s+final\s+answer\s+is\s*"
    r"\$?(.*?)\$?"
    r"\s*\.?\s*I\s+hope",
    re.DOTALL,
)
BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def extract_answer(text):
    """Extract answer from model output. Tries 'Final Answer' pattern first, then \\boxed{}."""
    # Try "Final Answer: The final answer is ..." pattern
    match = FINAL_ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    # Fallback: last \boxed{...}
    matches = BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return None


# --- Answer normalization (Minerva paper, Section D.1) ---

SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""),
    (r"\ ", ""), (" ", ""), ("mbox", "text"),
    (",\\text{and}", ","), ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "ft",
    "hours", "km", "units", "\\dots", "sue", "points", "feet",
    "minutes", "digits", "cents", "degrees", "cm", "gm", "pounds",
    "meters", "meals", "edges", "students", "childrentickets", "multiples",
    "\\text{s}", "\\text{.}", "\\text{\\ns}", "\\text{}^2",
    "\\text{}^3", "\\text{\\n}", "\\text{}", r"\mathrm{th}",
    r"^\circ", r"^{\circ}", r"\$", ",\\!", '"',
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold, is surrounded by a box, etc.
    final_answer = re.sub(r"(.*)\$(.+)\$(.*)", r"\2", final_answer)
    final_answer = re.sub(r"\\text\{(.*?)\}", r"\1", final_answer)
    final_answer = re.sub(r"\\textbf\{(.*?)\}", r"\1", final_answer)
    final_answer = re.sub(r"\\overline\{(.*?)\}", r"\1", final_answer)
    final_answer = re.sub(r"\\boxed\{(.*?)\}", r"\1", final_answer)

    # Normalize shorthand TeX
    final_answer = re.sub(r"\\frac\{(.*?)\}\{(.*?)\}", r"\\frac{\1}{\2}", final_answer)
    final_answer = re.sub(r"\\sqrt\{(.*?)\}", r"\\sqrt{\1}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize comma-separated numbers (100,000 -> 100000)
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_equiv(pred: str, ref: str) -> bool:
    """Compare two answers after normalization."""
    return normalize_final_answer(pred) == normalize_final_answer(ref)


# --- Task class ---

class MATH500OpenRouter:

    def __init__(self, model, n_shot=4, max_tokens=512, temperature=0.0, reasoning=False, log_dir=None):
        self.model = model
        self.n_shot = n_shot
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning = reasoning
        self.log_dir = log_dir
        self._log_lock = threading.Lock()
        self._log_file = None

        self.test_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

        effective_n = min(n_shot, len(FEW_SHOT_EXAMPLES))
        if n_shot > len(FEW_SHOT_EXAMPLES):
            print(f"Warning: MATH-500 only has {len(FEW_SHOT_EXAMPLES)} canonical few-shot examples, using {len(FEW_SHOT_EXAMPLES)} instead of {n_shot}")
        self.few_shot_examples = FEW_SHOT_EXAMPLES[:effective_n]

    def build_messages(self, question):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for fs_problem, fs_solution in self.few_shot_examples:
            messages.append({"role": "user", "content": f"Problem:\n{fs_problem}"})
            messages.append({"role": "assistant", "content": f"Solution:\n{fs_solution}"})
        messages.append({"role": "user", "content": f"Problem:\n{question}\n\n{INSTRUCTION}"})
        return messages

    def _log_result(self, record):
        if self._log_file is None:
            return
        with self._log_lock:
            self._log_file.write(json.dumps(record) + "\n")
            self._log_file.flush()

    def _eval_single(self, i):
        row = self.test_ds[i]
        question = row["problem"]
        ref_answer = row["answer"]
        subject = row["subject"]
        level = row["level"]
        messages = self.build_messages(question)
        try:
            response = chat(
                messages, model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                reasoning=self.reasoning,
            )
            completion = response["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"\nError on problem {i}: {e}")
            completion = ""
        pred_answer = extract_answer(completion)
        is_correct = (pred_answer is not None) and is_equiv(pred_answer, ref_answer)
        self._log_result({
            "index": i,
            "question": question,
            "subject": subject,
            "level": level,
            "ground_truth": ref_answer,
            "predicted": pred_answer,
            "correct": is_correct,
            "raw_response": completion,
        })
        return i, is_correct

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
                print(f"\r\033[K{num_correct}/{total} ({100*num_correct/total:.2f}%)", end="", flush=True)

        if self._log_file:
            self._log_file.close()
            self._log_file = None

        print()
        print("=" * 50)
        accuracy = num_correct / total if total > 0 else 0.0
        print(f"maj@1: {num_correct}/{total} ({100*accuracy:.2f}%)")
        return accuracy
