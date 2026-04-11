"""
GPQA Diamond evaluation via OpenRouter API.
Supports 4 prompting modes: 0-shot, 0-shot-cot, few-shot, few-shot-cot.
Based on prompts from arXiv 2311.12022, Appendix A.3.1.
"""

import os
import json
import re
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from open_router_tasks.common import chat
from open_router_tasks.gpqa_diamond_prompt import (
    ZERO_SHOT_PROMPT,
    ZERO_SHOT_COT_PROMPT,
    ZERO_SHOT_COT_EXTRACTION,
    FEW_SHOT_PREAMBLE,
    FEW_SHOT_COT_EXAMPLES,
    FEW_SHOT_FINAL_INSTRUCTION_COT,
    FEW_SHOT_FINAL_INSTRUCTION_NO_COT,
)

SYSTEM_PROMPT = "You are an expert in physics, chemistry, and biology."

VALID_MODES = ("0-shot", "0-shot-cot", "few-shot", "few-shot-cot")

LABELS = ["(A)", "(B)", "(C)", "(D)"]

# --- Answer extraction ---

ANSWER_RE = re.compile(r"correct answer is\s*\(?([A-D])\)?", re.IGNORECASE)
FALLBACK_RE = re.compile(r"\(([A-D])\)")


def extract_answer(text):
    """Extract (A)/(B)/(C)/(D) from model output."""
    match = ANSWER_RE.search(text)
    if match:
        return f"({match.group(1).upper()})"
    # Fallback: last standalone (A)/(B)/(C)/(D)
    matches = FALLBACK_RE.findall(text)
    if matches:
        return f"({matches[-1].upper()})"
    return None


def _shuffle_choices(row):
    """Deterministically shuffle choices for a question. Returns (choices_list, correct_label)."""
    correct = row["Correct Answer"]
    incorrects = [row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    choices = [correct] + incorrects
    # Deterministic seed from question text
    seed = hash(row["Question"]) & 0xFFFFFFFF
    rng = random.Random(seed)
    rng.shuffle(choices)
    correct_idx = choices.index(correct)
    return choices, LABELS[correct_idx]


def _format_choices(choices):
    """Format choices as (A) ... (B) ... etc."""
    return "\n".join(f"{LABELS[i]} {choices[i].strip()}" for i in range(4))


# --- Task class ---

class GPQADiamondOpenRouter:

    def __init__(self, model, mode="0-shot-cot", max_tokens=1024, temperature=0.0,
                 reasoning=False, log_dir=None):
        assert mode in VALID_MODES, f"Invalid mode: {mode}. Choose from {VALID_MODES}"
        self.model = model
        self.mode = mode
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning = reasoning
        self.log_dir = log_dir
        self._log_lock = threading.Lock()
        self._log_file = None

        self.test_ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")

    def _build_few_shot_messages(self, question_text, choices, is_cot):
        """Build few-shot messages with examples as user/assistant turns."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": FEW_SHOT_PREAMBLE})
        messages.append({"role": "assistant", "content": "I'll analyze each question carefully."})

        for ex_question, ex_reasoning_and_answer in FEW_SHOT_COT_EXAMPLES:
            messages.append({"role": "user", "content": ex_question})
            if is_cot:
                messages.append({"role": "assistant", "content": ex_reasoning_and_answer})
            else:
                # Strip reasoning, keep only "The correct answer is ..."
                lines = ex_reasoning_and_answer.strip().split("\n")
                answer_line = lines[-1]  # last line is always "The correct answer is (X)"
                messages.append({"role": "assistant", "content": answer_line})

        # Final question
        choices_text = _format_choices(choices)
        instruction = FEW_SHOT_FINAL_INSTRUCTION_COT if is_cot else FEW_SHOT_FINAL_INSTRUCTION_NO_COT
        final_prompt = f"Question: {question_text}\nChoices:\n{choices_text}\n\n{instruction}"
        messages.append({"role": "user", "content": final_prompt})
        return messages

    def build_messages(self, row):
        """Build chat messages for a given row based on the mode."""
        choices, correct_label = _shuffle_choices(row)
        question = row["Question"]

        if self.mode == "0-shot":
            prompt = ZERO_SHOT_PROMPT.format(
                question=question,
                choice_a=choices[0].strip(),
                choice_b=choices[1].strip(),
                choice_c=choices[2].strip(),
                choice_d=choices[3].strip(),
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

        elif self.mode == "0-shot-cot":
            prompt = ZERO_SHOT_COT_PROMPT.format(
                question=question,
                choice_a=choices[0].strip(),
                choice_b=choices[1].strip(),
                choice_c=choices[2].strip(),
                choice_d=choices[3].strip(),
            )
            # Single-turn: model produces reasoning, then we append extraction instruction
            full_prompt = prompt + "\n\n" + ZERO_SHOT_COT_EXTRACTION
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt},
            ]

        elif self.mode == "few-shot":
            messages = self._build_few_shot_messages(question, choices, is_cot=False)

        elif self.mode == "few-shot-cot":
            messages = self._build_few_shot_messages(question, choices, is_cot=True)

        return messages, correct_label

    def _log_result(self, record):
        if self._log_file is None:
            return
        with self._log_lock:
            self._log_file.write(json.dumps(record) + "\n")
            self._log_file.flush()

    def _eval_single(self, i):
        row = self.test_ds[i]
        messages, correct_label = self.build_messages(row)
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
        is_correct = (pred_answer is not None) and (pred_answer == correct_label)
        self._log_result({
            "index": i,
            "question": row["Question"],
            "domain": row.get("High-level domain", ""),
            "subdomain": row.get("Subdomain", ""),
            "ground_truth": correct_label,
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
            log_path = os.path.join(self.log_dir, f"{self.model.replace('/', '_')}_{self.mode}.jsonl")
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
