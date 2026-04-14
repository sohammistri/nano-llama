"""
ARC Challenge evaluation via OpenRouter API.
0-shot prompting with letter-answer extraction.
"""

import json
import re
import random
from datasets import load_dataset
from open_router_tasks.common import BaseOpenRouterTask

SYSTEM_PROMPT = "You are an expert at science and logical reasoning."

LABELS = ["A", "B", "C", "D", "E"]
COUNT_WORDS = {3: "three", 4: "four", 5: "five"}

ANSWER_RE = re.compile(r"best answer is\s*\(?([A-E])\)?", re.IGNORECASE)
FALLBACK_RE = re.compile(r"\(([A-E])\)")


def extract_answer(text):
    """Extract answer letter from model output."""
    match = ANSWER_RE.search(text)
    if match:
        return match.group(1).upper()
    matches = FALLBACK_RE.findall(text)
    if matches:
        return matches[-1].upper()
    return None


def _normalize_row(row):
    """Normalize a dataset row so choices always use A/B/C/D/E labels.
    Handles both letter and numeric label schemes via index lookup.
    Returns (question, texts, labels, correct_label)."""
    question = row["question"]
    texts = row["choices"]["text"]
    orig_labels = row["choices"]["label"]
    answer_key = row["answerKey"]
    correct_idx = orig_labels.index(answer_key)
    n = len(texts)
    labels = LABELS[:n]
    correct_label = labels[correct_idx]
    return question, texts, labels, correct_label


class ARCChallengeOpenRouter(BaseOpenRouterTask):

    def __init__(self, model, max_tokens=2**16, temperature=0.0, reasoning=False, log_dir=None):
        super().__init__(model, max_tokens=max_tokens, temperature=temperature, reasoning=reasoning, log_dir=log_dir)
        self.test_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

    def _log_filename(self):
        return f"{self.model.replace('/', '_')}_arc_challenge.jsonl"

    def build_messages(self, row):
        question, texts, labels, correct_label = _normalize_row(row)
        n = len(labels)

        choices_str = "\n".join(f"{labels[i]}. {texts[i]}" for i in range(n))
        count_word = COUNT_WORDS.get(n, str(n))
        label_list = ", ".join(labels[:-1]) + f" and {labels[-1]}"
        label_options = ", ".join(labels[:-1]) + f" or {labels[-1]}"

        user_content = (
            f"Given the following question and {count_word} candidate answers "
            f"({label_list}), choose the best answer.\n"
            f"Question: {question}\n"
            f"{choices_str}\n"
            f'Your response should end with "The best answer is [the_answer_letter]" '
            f"where the [the_answer_letter] is one of {label_options}."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
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
            "id": row["id"],
            "question": row["question"],
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
        _, texts, labels, _ = _normalize_row(row)
        choices_str = "\n".join(f"{labels[j]}. {texts[j]}" for j in range(len(labels)))

        print(SEP)
        print(f"DEBUG: ARC_CHALLENGE  |  Problem #{i} of {len(self.test_ds)}")
        print(SEP)
        print(f"\n[QUESTION]\n{row['question']}")
        print(f"\n[CHOICES]\n{choices_str}")
        print(f"\n[GROUND TRUTH]  {correct_label}")
        print(f"\n[PAYLOAD -- {len(messages)} messages]")
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
