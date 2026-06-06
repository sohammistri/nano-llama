"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import json
import re
import time
from urllib.parse import quote
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanollama.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The HuggingFace dataset repo where the default data is hosted and downloaded from on demand
DEFAULT_DATASET_HANDLE = "karpathy/climbmix-400b-shuffle"
DEFAULT_REVISION = "main"
DEFAULT_TEXT_COLUMN = "text"
DATASET_HANDLE_ENV = "NANOLLAMA_DATASET_HANDLE"
TEXT_COLUMN_ENV = "NANOLLAMA_TEXT_COLUMN"
BASE_URL = f"https://huggingface.co/datasets/{DEFAULT_DATASET_HANDLE}/resolve/{DEFAULT_REVISION}"
MAX_SHARD = 6542 # the last datashard is shard_06542.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")

def get_dataset_handle(dataset_handle=None):
    return dataset_handle or os.environ.get(DATASET_HANDLE_ENV, DEFAULT_DATASET_HANDLE)

def dataset_data_dir(dataset_handle):
    """Return the local cache directory for a HuggingFace dataset handle."""
    dataset_handle = get_dataset_handle(dataset_handle)
    if dataset_handle == DEFAULT_DATASET_HANDLE:
        return DATA_DIR
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", dataset_handle).strip("_")
    return os.path.join(base_dir, f"base_data_{safe_name}")

def dataset_metadata_path(data_dir):
    return os.path.join(data_dir, ".nanollama_dataset.json")

def write_dataset_metadata(data_dir, dataset_handle, revision, text_column):
    metadata = {
        "dataset_handle": dataset_handle,
        "revision": revision,
        "text_column": text_column,
    }
    with open(dataset_metadata_path(data_dir), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")

def read_dataset_metadata(data_dir):
    path = dataset_metadata_path(data_dir)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

def get_text_column(text_column=None, data_dir=None):
    if text_column is not None:
        return text_column
    if os.environ.get(TEXT_COLUMN_ENV):
        return os.environ[TEXT_COLUMN_ENV]
    if data_dir is not None:
        metadata = read_dataset_metadata(data_dir)
        if metadata.get("text_column"):
            return metadata["text_column"]
    return DEFAULT_TEXT_COLUMN

def hf_dataset_resolve_url(dataset_handle, revision, remote_path):
    quoted_path = quote(remote_path, safe="/")
    return f"https://huggingface.co/datasets/{dataset_handle}/resolve/{revision}/{quoted_path}"

def list_hf_parquet_files(dataset_handle, revision=DEFAULT_REVISION):
    """List parquet files in a HuggingFace dataset repo."""
    api_url = f"https://huggingface.co/api/datasets/{dataset_handle}/tree/{revision}"
    params = {"recursive": "true", "limit": 1000}
    parquet_files = []
    discovered_files = []
    while api_url is not None:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        entries = response.json()
        for entry in entries:
            path = entry.get("path", "")
            if entry.get("type") == "file":
                discovered_files.append(path)
                if path.endswith(".parquet"):
                    parquet_files.append(path)
        api_url = response.links.get("next", {}).get("url")
        params = None

    parquet_files = sorted(parquet_files)
    if not parquet_files:
        suffixes = sorted({os.path.splitext(path)[1] or "<no extension>" for path in discovered_files})
        suffix_summary = ", ".join(suffixes[:10]) if suffixes else "no files"
        raise RuntimeError(
            f"No parquet files found in HuggingFace dataset {dataset_handle}@{revision}. "
            f"This downloader expects parquet shards. Found file types: {suffix_summary}"
        )
    return parquet_files

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None, warn_on_legacy=False, dataset_handle=None):
    """ Looks into a data dir and returns full paths to all parquet files. """
    dataset_handle = get_dataset_handle(dataset_handle)
    data_dir = dataset_data_dir(dataset_handle) if data_dir is None else data_dir

    # Legacy-supporting code due to the upgrade from FinewebEdu-100B to ClimbMix-400B
    # This code will eventually be deleted.
    if dataset_handle == DEFAULT_DATASET_HANDLE and not os.path.exists(data_dir):
        if warn_on_legacy:
            print()
            print("=" * 80)
            print("  WARNING: DATASET UPGRADE REQUIRED")
            print("=" * 80)
            print()
            print(f"  Could not find: {data_dir}")
            print()
            print("  nanollama recently switched from FinewebEdu-100B to ClimbMix-400B.")
            print("  Everyone who does `git pull` as of March 4, 2026 is expected to see this message.")
            print("  To upgrade to the new ClimbMix-400B dataset, run these two commands:")
            print()
            print("    python -m nanollama.dataset -n 170     # download ~170 shards, enough for GPT-2, adjust as desired")
            print("    python -m scripts.tok_train           # re-train tokenizer on new ClimbMix data")
            print()
            print("  For now, falling back to your old FinewebEdu-100B dataset...")
            print("=" * 80)
            print()
        # attempt a fallback to the legacy data directory
        data_dir = os.path.join(base_dir, "base_data")

    parquet_paths = []
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.parquet') and not filename.endswith('.tmp'):
                parquet_paths.append(os.path.join(root, filename))
    parquet_paths = sorted(parquet_paths)
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1, dataset_handle=None, text_column=None):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    dataset_handle = get_dataset_handle(dataset_handle)
    data_dir = dataset_data_dir(dataset_handle)
    text_column = get_text_column(text_column, data_dir=data_dir)
    parquet_paths = list_parquet_files(data_dir=data_dir, dataset_handle=dataset_handle)
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column(text_column).to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def download_single_file(file_spec):
    """ Downloads a single file index, with some backoff """

    if isinstance(file_spec, int):
        dataset_handle = DEFAULT_DATASET_HANDLE
        revision = DEFAULT_REVISION
        remote_path = index_to_filename(file_spec)
        data_dir = DATA_DIR
    else:
        dataset_handle, revision, remote_path, data_dir = file_spec

    # Construct the local filepath for this file and skip if it already exists
    filepath = os.path.join(data_dir, remote_path)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = hf_dataset_resolve_url(dataset_handle, revision, remote_path)
    print(f"Downloading {remote_path}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {remote_path}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {remote_path}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {remote_path} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of train shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    parser.add_argument("-d", "--dataset-handle", type=str, default=DEFAULT_DATASET_HANDLE, help=f"HuggingFace dataset handle to download from (default: {DEFAULT_DATASET_HANDLE})")
    parser.add_argument("-r", "--revision", type=str, default=DEFAULT_REVISION, help=f"HuggingFace dataset revision to download from (default: {DEFAULT_REVISION})")
    parser.add_argument("--text-column", type=str, default=DEFAULT_TEXT_COLUMN, help=f"Parquet column containing document text (default: {DEFAULT_TEXT_COLUMN}; use content for tiiuae/falcon-refinedweb)")
    args = parser.parse_args()

    # Prepare the output directory
    data_dir = dataset_data_dir(args.dataset_handle)
    os.makedirs(data_dir, exist_ok=True)
    write_dataset_metadata(data_dir, args.dataset_handle, args.revision, args.text_column)

    # The way this works is that the user specifies the number of train shards to download via the -n flag.
    # In addition to that, the validation shard is *always* downloaded and is pinned to be the last shard.
    parquet_files = list_hf_parquet_files(args.dataset_handle, args.revision)
    print(f"Found {len(parquet_files)} parquet shards in {args.dataset_handle}@{args.revision}")
    num_train_shards = len(parquet_files) - 1 if args.num_files == -1 else min(args.num_files, len(parquet_files) - 1)
    files_to_download = parquet_files[:num_train_shards]
    files_to_download.append(parquet_files[-1]) # always download the validation shard
    files_to_download = sorted(set(files_to_download))
    file_specs = [(args.dataset_handle, args.revision, remote_path, data_dir) for remote_path in files_to_download]

    # Download the shards
    print(f"Downloading {len(file_specs)} shards using {args.num_workers} workers...")
    print(f"Target directory: {data_dir}")
    print(f"Text column: {args.text_column}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, file_specs)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(file_specs)} shards to {data_dir}")
