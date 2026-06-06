"""
Upload locally saved W&B offline runs to the W&B cloud.

Examples:
    uv run python -m scripts.upload_logs_to_wandb --run my-run
    uv run python -m scripts.upload_logs_to_wandb --run my-run --project nanollama --entity my-team
    uv run python -m scripts.upload_logs_to_wandb ~/.cache/nanollama/logs/my-run/wandb/offline-run-*
    uv run python -m scripts.upload_logs_to_wandb --dry-run --all
"""

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path


DEFAULT_LOGS_ROOT = Path("~/.cache/nanollama/logs").expanduser()


def expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def is_wandb_run_path(path: Path) -> bool:
    return path.is_dir() and path.name.startswith(("offline-run-", "run-"))


def find_wandb_runs(path: Path) -> list[Path]:
    """Find W&B run directories under a path.

    Accepts a direct offline-run directory, a W&B directory, a nano-llama run
    directory, or the full logs root.
    """
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    if is_wandb_run_path(path):
        return [path]

    candidates = []
    candidates.extend(path.glob("offline-run-*"))
    candidates.extend(path.glob("run-*"))
    candidates.extend(path.glob("wandb/offline-run-*"))
    candidates.extend(path.glob("wandb/run-*"))
    candidates.extend(path.glob("*/wandb/offline-run-*"))
    candidates.extend(path.glob("*/wandb/run-*"))

    return sorted({candidate.resolve() for candidate in candidates if is_wandb_run_path(candidate)})


def discover_runs(logs_root: Path, run_names: list[str], paths: list[str], sync_all: bool) -> list[Path]:
    search_paths = []
    if paths:
        search_paths.extend(expand_path(path) for path in paths)
    elif run_names:
        search_paths.extend(logs_root / run_name for run_name in run_names)
    elif sync_all:
        search_paths.append(logs_root)
    else:
        raise SystemExit("Specify --run RUN, pass one or more paths, or use --all.")

    discovered = []
    for search_path in search_paths:
        runs = find_wandb_runs(search_path)
        if not runs:
            print(f"WARNING: no W&B offline runs found under {search_path}")
        discovered.extend(runs)

    return sorted({run.resolve() for run in discovered})


def build_wandb_sync_command(args: argparse.Namespace, run_paths: list[Path]) -> list[str]:
    wandb_executable = shutil.which("wandb")
    if wandb_executable is None:
        raise SystemExit(
            "Could not find the 'wandb' executable. Run this script with "
            "`uv run python -m scripts.upload_logs_to_wandb ...`."
        )

    command = [wandb_executable, "sync", "--project", args.project]
    if args.entity:
        command.extend(["--entity", args.entity])
    if args.job_type:
        command.extend(["--job_type", args.job_type])
    if args.include_synced:
        command.append("--include-synced")
    if not args.mark_synced:
        command.append("--no-mark-synced")
    if args.append:
        command.append("--append")
    if args.skip_console:
        command.append("--skip-console")
    if args.replace_tags:
        command.extend(["--replace-tags", args.replace_tags])
    command.extend(str(path) for path in run_paths)
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload nano-llama W&B offline logs to the W&B cloud")
    parser.add_argument(
        "paths",
        nargs="*",
        help="explicit W&B offline-run directories, W&B dirs, nano-llama run dirs, or a logs root",
    )
    parser.add_argument("--logs-root", type=expand_path, default=DEFAULT_LOGS_ROOT, help="root directory containing local runs")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="run name under --logs-root to upload; repeat to upload several runs",
    )
    parser.add_argument("--all", action="store_true", help="upload all discovered W&B offline runs under --logs-root")
    parser.add_argument("--project", type=str, default="nanollama", help="W&B project to upload to")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity/team/user to upload to")
    parser.add_argument("--job-type", type=str, default=None, help="optional W&B job type")
    parser.add_argument("--include-synced", action="store_true", help="include runs already marked as synced")
    parser.add_argument(
        "--no-mark-synced",
        dest="mark_synced",
        action="store_false",
        help="do not mark runs as synced after upload",
    )
    parser.add_argument("--append", action="store_true", help="append to an existing W&B run if applicable")
    parser.add_argument("--skip-console", action="store_true", help="skip uploading console logs")
    parser.add_argument("--replace-tags", type=str, default=None, help="tag replacements, e.g. old_tag=new_tag,foo=bar")
    parser.add_argument("--dry-run", action="store_true", help="print the wandb sync command without running it")
    parser.set_defaults(mark_synced=True)
    args = parser.parse_args()

    run_paths = discover_runs(args.logs_root, args.run, args.paths, args.all)
    if not run_paths:
        raise SystemExit("No W&B offline runs found.")

    print(f"Found {len(run_paths)} W&B run(s):")
    for run_path in run_paths:
        print(f"  {run_path}")

    command = build_wandb_sync_command(args, run_paths)
    print(f"\nCommand:\n  {shlex.join(command)}")
    if args.dry_run:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()