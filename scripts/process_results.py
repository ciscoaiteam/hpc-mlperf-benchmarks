#!/usr/bin/env python3
"""
process_results.py
==================
Parse MLPerf Training benchmark logs and compute official statistics.

MLPerf closed-division rule:
  - Run each config 10 times
  - Discard the single fastest and single slowest run
  - Average the remaining 8

Usage:
  # Process a single run directory:
  python3 process_results.py /data/mlperf/logs/llama31_pretraining/llama31_pretraining_8xH200_20250414_120000

  # Process all logs under a root dir and append to a summary file:
  python3 process_results.py --log-root /data/mlperf/logs --output summary.txt --append
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ---------------------------------------------------------------------------
# MLPerf MLLOG parser
# ---------------------------------------------------------------------------
MLLOG_RE = re.compile(r":::MLLOG\s+(\{.*\})")

def parse_mllog(log_path: Path) -> dict:
    """
    Extract run_start / run_stop timestamps and final eval accuracy from
    an MLPerf MLLOG-formatted log file.

    Returns dict with keys: status, wall_time_ms, eval_value
    """
    run_start_ms: Optional[float] = None
    run_stop_ms: Optional[float] = None
    status = "unknown"
    eval_value: Optional[float] = None

    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                m = MLLOG_RE.search(line)
                if not m:
                    continue
                try:
                    entry = json.loads(m.group(1))
                except json.JSONDecodeError:
                    continue

                key = entry.get("key", "")
                ts  = entry.get("time_ms")
                val = entry.get("value")
                meta = entry.get("metadata", {}) or {}

                if key == "run_start" and ts is not None:
                    run_start_ms = float(ts)
                elif key == "run_stop" and ts is not None:
                    run_stop_ms = float(ts)
                    status = meta.get("status", "unknown")
                elif key in ("eval_map", "eval_accuracy", "eval_loss"):
                    if val is not None:
                        eval_value = float(val)
    except FileNotFoundError:
        pass

    wall_time_ms = None
    if run_start_ms is not None and run_stop_ms is not None:
        wall_time_ms = run_stop_ms - run_start_ms

    return {
        "status": status,
        "wall_time_ms": wall_time_ms,
        "wall_time_min": wall_time_ms / 60_000 if wall_time_ms else None,
        "eval_value": eval_value,
    }


def parse_times_file(times_path: Path) -> list[dict]:
    """
    Fallback: parse the wall_times.txt written by the run scripts if MLLOG
    entries are missing.

    Format: run_num, status, wall_time_min
    """
    results = []
    try:
        with open(times_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    results.append({
                        "run": int(parts[0]),
                        "status": parts[1],
                        "wall_time_min": float(parts[2]),
                    })
    except FileNotFoundError:
        pass
    return results


# ---------------------------------------------------------------------------
# Statistics (MLPerf closed-division)
# ---------------------------------------------------------------------------
def compute_mlperf_stats(times_min: list[float]) -> dict:
    """
    MLPerf rule: drop fastest + slowest, average the rest.
    Requires >= 5 successful runs to be reportable.
    """
    n = len(times_min)
    if n < 2:
        return {"error": f"only {n} successful run(s); need at least 5"}

    sorted_times = sorted(times_min)
    trimmed = sorted_times[1:-1]  # drop min and max

    if HAS_NUMPY:
        mean  = float(np.mean(trimmed))
        stdev = float(np.std(trimmed, ddof=1)) if len(trimmed) > 1 else 0.0
    else:
        mean  = sum(trimmed) / len(trimmed)
        stdev = 0.0

    return {
        "n_total": n,
        "n_success": n,
        "n_trimmed": len(trimmed),
        "min_min": sorted_times[0],
        "max_min": sorted_times[-1],
        "dropped_fastest_min": sorted_times[0],
        "dropped_slowest_min": sorted_times[-1],
        "trimmed_mean_min": mean,
        "trimmed_stdev_min": stdev,
    }


# ---------------------------------------------------------------------------
# Process a single run directory
# ---------------------------------------------------------------------------
def process_run_dir(run_dir: Path, verbose: bool = True) -> Optional[dict]:
    """
    Collect results from all run_N.log files and wall_times.txt.
    Returns aggregated stats dict or None if no data found.
    """
    times_file = run_dir / "wall_times.txt"
    rows = parse_times_file(times_file)

    # Try to enrich/override with MLLOG data from individual run logs
    mllog_rows = []
    for log_file in sorted(run_dir.glob("run_*.log")):
        # skip docker sub-logs
        if "_docker.log" in log_file.name:
            continue
        m = re.match(r"run_(\d+)\.log", log_file.name)
        if not m:
            continue
        run_num = int(m.group(1))
        parsed = parse_mllog(log_file)
        if parsed["wall_time_min"] is not None:
            mllog_rows.append({
                "run": run_num,
                "status": parsed["status"],
                "wall_time_min": parsed["wall_time_min"],
                "eval_value": parsed["eval_value"],
            })

    # Prefer MLLOG data; fall back to wall_times.txt
    all_rows = mllog_rows if mllog_rows else rows
    if not all_rows:
        return None

    # Filter to successful runs only
    success_rows = [r for r in all_rows if r["status"] == "success"]
    failed_count = len(all_rows) - len(success_rows)
    success_times = [r["wall_time_min"] for r in success_rows]

    stats = compute_mlperf_stats(success_times)
    stats["run_dir"] = str(run_dir)
    stats["n_total_attempted"] = len(all_rows)
    stats["n_failed"] = failed_count

    # Infer benchmark name and GPU count from directory name
    dir_name = run_dir.name
    stats["tag"] = dir_name

    m_bench = re.search(r"(llama31_pretraining|llm_finetuning)", dir_name)
    m_gpu   = re.search(r"(\d+)xH200", dir_name)
    stats["benchmark"] = m_bench.group(1) if m_bench else "unknown"
    stats["num_gpus"]  = int(m_gpu.group(1)) if m_gpu else 0

    if verbose:
        _print_stats(stats)

    return stats


def _print_stats(s: dict):
    bench = s.get("benchmark", "?")
    ngpu  = s.get("num_gpus", "?")
    print(f"\n{'='*60}")
    print(f"  Benchmark : {bench}")
    print(f"  GPUs      : {ngpu}x H200")
    print(f"  Tag       : {s.get('tag', '')}")
    print(f"{'='*60}")
    if "error" in s:
        print(f"  ERROR: {s['error']}")
        return
    print(f"  Runs attempted      : {s['n_total_attempted']}")
    print(f"  Successful          : {s['n_success']}")
    print(f"  Failed/aborted      : {s['n_failed']}")
    print(f"  Fastest (dropped)   : {s['dropped_fastest_min']:.2f} min")
    print(f"  Slowest (dropped)   : {s['dropped_slowest_min']:.2f} min")
    print(f"  Trimmed runs used   : {s['n_trimmed']}")
    print(f"  >> Mean wall time   : {s['trimmed_mean_min']:.2f} min  "
          f"(±{s['trimmed_stdev_min']:.2f})")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Process MLPerf Training benchmark results"
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        help="Path to a single run directory (e.g. .../retinanet_8xH200_20250414_120000)"
    )
    parser.add_argument(
        "--log-root",
        help="Root logs directory; all sub-directories will be scanned"
    )
    parser.add_argument(
        "--output",
        help="Write summary to this file"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to --output instead of overwriting"
    )
    args = parser.parse_args()

    all_stats = []

    if args.run_dir:
        s = process_run_dir(Path(args.run_dir))
        if s:
            all_stats.append(s)

    elif args.log_root:
        root = Path(args.log_root)
        # Find all leaf directories that contain wall_times.txt or run_*.log
        for candidate in sorted(root.rglob("wall_times.txt")):
            d = candidate.parent
            s = process_run_dir(d)
            if s:
                all_stats.append(s)
    else:
        parser.print_help()
        sys.exit(1)

    if not all_stats:
        print("No results found.")
        sys.exit(0)

    # ---------------------------------------------------------------------------
    # Print consolidated table
    # ---------------------------------------------------------------------------
    print("\n\n" + "="*80)
    print("  MLPERF TRAINING CONSOLIDATED RESULTS (H200)")
    print("="*80)
    header = f"{'Benchmark':<22} {'GPUs':>5} {'Runs':>5} {'Mean (min)':>12} {'±Stdev':>8}"
    print(header)
    print("-"*80)
    for s in sorted(all_stats, key=lambda x: (x["benchmark"], x["num_gpus"])):
        if "error" in s:
            print(f"  {s['benchmark']:<20} {s['num_gpus']:>5}  ERROR: {s['error']}")
        else:
            print(
                f"  {s['benchmark']:<20} {s['num_gpus']:>5} "
                f"{s['n_trimmed']:>5} "
                f"{s['trimmed_mean_min']:>12.2f} "
                f"{s['trimmed_stdev_min']:>8.2f}"
            )
    print("="*80)
    print("  Note: Mean is trimmed (fastest + slowest run dropped per MLPerf rules)")
    print("="*80)

    # ---------------------------------------------------------------------------
    # Write to file if requested
    # ---------------------------------------------------------------------------
    if args.output:
        mode = "a" if args.append else "w"
        with open(args.output, mode) as f:
            f.write("\n\nMLPERF TRAINING CONSOLIDATED RESULTS\n")
            f.write(f"Generated: {__import__('datetime').datetime.now()}\n\n")
            f.write(f"{'Benchmark':<22} {'GPUs':>5} {'Runs':>5} "
                    f"{'Mean (min)':>12} {'±Stdev':>8}\n")
            f.write("-"*60 + "\n")
            for s in sorted(all_stats, key=lambda x: (x["benchmark"], x["num_gpus"])):
                if "error" in s:
                    f.write(f"{s['benchmark']:<22} {s['num_gpus']:>5}  "
                            f"ERROR: {s['error']}\n")
                else:
                    f.write(
                        f"{s['benchmark']:<22} {s['num_gpus']:>5} "
                        f"{s['n_trimmed']:>5} "
                        f"{s['trimmed_mean_min']:>12.2f} "
                        f"{s['trimmed_stdev_min']:>8.2f}\n"
                    )
        print(f"\nResults appended to: {args.output}")


if __name__ == "__main__":
    main()
