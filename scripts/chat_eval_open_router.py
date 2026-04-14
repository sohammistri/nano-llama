"""
Evaluate models via OpenRouter API on benchmark tasks.

Example:
python -m scripts.chat_eval_open_router -m meta-llama/llama-3.2-3b-instruct -a GSM8K -n 5 -x 20
"""

import os
import json
import argparse
from datetime import datetime
from open_router_tasks.gsm8k import GSM8KOpenRouter
from open_router_tasks.math500 import MATH500OpenRouter
from open_router_tasks.gpqa_diamond import GPQADiamondOpenRouter
from open_router_tasks.arc_challenge import ARCChallengeOpenRouter
from open_router_tasks.mbppplus import MBPPPlusOpenRouter
from open_router_tasks.humaneval_plus import HumanEvalPlusOpenRouter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models via OpenRouter API")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="OpenRouter model name (e.g. meta-llama/llama-3.2-3b-instruct)")
    parser.add_argument('-a', '--task-name', type=str, nargs='+', default=["GSM8K"],
                        help="Task name(s) (GSM8K, MATH500, GPQA_DIAMOND, ARC_CHALLENGE, MBPP_PLUS, HUMANEVAL_PLUS, ALL)")
    parser.add_argument('--gpqa-mode', type=str, nargs='+', default=["0-shot-cot"],
                        help="Prompting mode(s) for GPQA_DIAMOND (0-shot, 0-shot-cot, few-shot, few-shot-cot, ALL)")
    parser.add_argument('-t', '--temperature', type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument('--max-tokens', type=int, default=2**16,
                        help="Max tokens for model response (default: 65536)")
    parser.add_argument('-x', '--max-problems', type=int, default=None,
                        help="Max problems to evaluate (default: all)")
    parser.add_argument('--reasoning', action='store_true', default=False,
                        help="Enable reasoning mode for supported models")
    parser.add_argument('-w', '--workers', type=int, default=10,
                        help="Number of concurrent API workers (default: 10)")
    parser.add_argument('--log', action='store_true', default=False,
                        help="Log responses to .cache/nanollama/<task>/ as JSONL")
    parser.add_argument('--debug', action='store_true', default=False,
                        help="Debug mode: pick one random problem, print payload, response, and grading")
    parser.add_argument('--debug-seed', type=int, default=None,
                        help="Random seed for --debug problem selection (default: random)")
    args = parser.parse_args()

    task_map = {
        'GSM8K': GSM8KOpenRouter,
        'MATH500': MATH500OpenRouter,
        'GPQA_DIAMOND': GPQADiamondOpenRouter,
        'ARC_CHALLENGE': ARCChallengeOpenRouter,
        'MBPP_PLUS': MBPPPlusOpenRouter,
        'HUMANEVAL_PLUS': HumanEvalPlusOpenRouter,
    }
    gpqa_modes = ['0-shot', '0-shot-cot', 'few-shot', 'few-shot-cot']
    task_names = list(task_map.keys()) if 'ALL' in args.task_name else args.task_name
    for tn in task_names:
        assert tn in task_map, f"Unknown task: {tn}. Available: {list(task_map.keys())}"

    results_dir = os.path.join(".cache", "nanollama", "results")
    os.makedirs(results_dir, exist_ok=True)
    model_slug = args.model.replace("/", "_")

    for tn in task_names:
        if tn == 'GPQA_DIAMOND':
            modes = gpqa_modes if 'ALL' in args.gpqa_mode else args.gpqa_mode
        else:
            modes = [None]
        for mode in modes:
            log_dir = None
            if args.log:
                log_dir = os.path.join(".cache", "nanollama", tn.lower())

            task_kwargs = dict(
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                reasoning=args.reasoning,
                log_dir=log_dir,
            )
            if tn == 'GPQA_DIAMOND':
                task_kwargs['mode'] = mode
            task = task_map[tn](**task_kwargs)

            print(f"Model: {args.model}")
            print(f"Task: {tn}")
            if tn == 'GPQA_DIAMOND':
                print(f"GPQA Mode: {mode}")
            print(f"Temperature: {args.temperature}")
            print(f"Reasoning: {args.reasoning}")
            print(f"Workers: {args.workers}")
            print(f"Logging: {args.log}")
            print("=" * 50)

            if args.debug:
                task.debug_single(seed=args.debug_seed)
                continue

            accuracy = task.run_eval(max_problems=args.max_problems, workers=args.workers)

            # Save results
            suffix = f"_{mode}" if tn == 'GPQA_DIAMOND' else ""
            results_path = os.path.join(results_dir, f"{model_slug}_{tn.lower()}{suffix}.json")
            results = {
                "model": args.model,
                "task": tn,
                "temperature": args.temperature,
                "reasoning": args.reasoning,
                "max_problems": args.max_problems,
                "accuracy": accuracy,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            if tn == 'GPQA_DIAMOND':
                results["gpqa_mode"] = mode
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {results_path}")
