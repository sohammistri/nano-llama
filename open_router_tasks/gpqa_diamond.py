"""
GPQA Diamond evaluation via OpenRouter API.
Supports 4 prompting modes: 0-shot, 0-shot-cot, few-shot, few-shot-cot.
Based on prompts from arXiv 2311.12022, Appendix A.3.1.
"""

import json
import re
import random
from datasets import load_dataset
from open_router_tasks.common import BaseOpenRouterTask
from open_router_tasks.gpqa_diamond_prompt import (
    ZERO_SHOT_PROMPT,
    ZERO_SHOT_COT_PROMPT,
    ZERO_SHOT_ANSWER_EXTRACTION,
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

ANSWER_RE = re.compile(r"best answer is\s*\(?([A-D])\)?", re.IGNORECASE)
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

class GPQADiamondOpenRouter(BaseOpenRouterTask):

    def __init__(self, model, mode="0-shot-cot", max_tokens=2**16, temperature=0.0,
                 reasoning=False, log_dir=None):
        assert mode in VALID_MODES, f"Invalid mode: {mode}. Choose from {VALID_MODES}"
        super().__init__(model, max_tokens=max_tokens, temperature=temperature, reasoning=reasoning, log_dir=log_dir)
        self.mode = mode

        self.test_ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")

    def _log_filename(self):
        return f"{self.model.replace('/', '_')}_{self.mode}.jsonl"

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
            full_prompt = prompt + "\n\n" + ZERO_SHOT_ANSWER_EXTRACTION
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt},
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

    def _eval_single(self, i):
        row = self.test_ds[i]
        messages, correct_label = self.build_messages(row)
        try:
            response = self._chat(messages)
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

    def debug_single(self, seed=None):
        """Run on one random problem, printing payload, response, and grading."""
        SEP = "=" * 80
        i = random.Random(seed).randrange(len(self.test_ds))
        row = self.test_ds[i]
        messages, correct_label = self.build_messages(row)

        # Re-derive choices for display (same deterministic shuffle)
        choices, _ = _shuffle_choices(row)
        choices_text = _format_choices(choices)

        print(SEP)
        print(f"DEBUG: GPQA_DIAMOND  |  Problem #{i} of {len(self.test_ds)}  "
              f"[mode={self.mode}]")
        print(f"Domain: {row.get('High-level domain', '')} / "
              f"{row.get('Subdomain', '')}")
        print(SEP)
        print(f"\n[QUESTION]\n{row['Question']}")
        print(f"\n[CHOICES]\n{choices_text}")
        print(f"\n[GROUND TRUTH]  {correct_label}")
        print(f"\n[PAYLOAD — {len(messages)} messages]")
        print(json.dumps(messages, indent=2))

        response = self._chat(messages)
        print(f"\n[API RESPONSE]")
        print(json.dumps(response, indent=2))

        completion = response["choices"][0]["message"]["content"]
        pred_answer = extract_answer(completion)
        is_correct = (pred_answer is not None) and (pred_answer == correct_label)

        print(f"\n[COMPLETION]\n{completion}")
        print(f"\n[EXTRACTED ANSWER]  {pred_answer}")
        print(f"\n[GRADING]  {'CORRECT' if is_correct else 'INCORRECT'}  "
              f"(pred={pred_answer}, ref={correct_label})")
        print(SEP)

