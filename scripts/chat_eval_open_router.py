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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models via OpenRouter API")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="OpenRouter model name (e.g. meta-llama/llama-3.2-3b-instruct)")
    parser.add_argument('-a', '--task-name', type=str, default="GSM8K",
                        help="Task name (currently: GSM8K, MATH500, GPQA_DIAMOND)")
    parser.add_argument('--mode', type=str, default="0-shot-cot",
                        help="Prompting mode for GPQA_DIAMOND (0-shot, 0-shot-cot, few-shot, few-shot-cot)")
    parser.add_argument('-t', '--temperature', type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument('-x', '--max-problems', type=int, default=None,
                        help="Max problems to evaluate (default: all)")
    parser.add_argument('--reasoning', action='store_true', default=False,
                        help="Enable reasoning mode for supported models")
    parser.add_argument('-w', '--workers', type=int, default=10,
                        help="Number of concurrent API workers (default: 10)")
    parser.add_argument('--log', action='store_true', default=False,
                        help="Log responses to .cache/nanollama/<task>/ as JSONL")
    args = parser.parse_args()

    task_map = {
        'GSM8K': GSM8KOpenRouter,
        'MATH500': MATH500OpenRouter,
        'GPQA_DIAMOND': GPQADiamondOpenRouter,
    }
    assert args.task_name in task_map, f"Unknown task: {args.task_name}. Available: {list(task_map.keys())}"

    log_dir = None
    if args.log:
        log_dir = os.path.join(".cache", "nanollama", args.task_name.lower())

    task_kwargs = dict(
        model=args.model,
        temperature=args.temperature,
        reasoning=args.reasoning,
        log_dir=log_dir,
    )
    if args.task_name == 'GPQA_DIAMOND':
        task_kwargs['mode'] = args.mode
    task = task_map[args.task_name](**task_kwargs)

    print(f"Model: {args.model}")
    print(f"Task: {args.task_name}")
    print(f"Temperature: {args.temperature}")
    print(f"Reasoning: {args.reasoning}")
    print(f"Workers: {args.workers}")
    print(f"Logging: {args.log}")
    print("=" * 50)

    accuracy = task.run_eval(max_problems=args.max_problems, workers=args.workers)

    # Save results
    results_dir = os.path.join(".cache", "nanollama", "results")
    os.makedirs(results_dir, exist_ok=True)
    model_slug = args.model.replace("/", "_")
    results_path = os.path.join(results_dir, f"{model_slug}_{args.task_name.lower()}.json")
    results = {
        "model": args.model,
        "task": args.task_name,
        "temperature": args.temperature,
        "reasoning": args.reasoning,
        "max_problems": args.max_problems,
        "accuracy": accuracy,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
