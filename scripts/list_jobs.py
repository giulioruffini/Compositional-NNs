#!/usr/bin/env python3
"""
List jobs from the central jobs registry.

Usage:
  python list_jobs.py [--registry PATH] [--last N] [--status success|failed|running]

Defaults: registry=../jobs/jobs_registry.json (when run from scripts/).
"""

import argparse
import json
import os
from pathlib import Path


def load_registry(path: str) -> list:
    """Load registry JSON; return list of job records."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p, 'r') as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def main():
    parser = argparse.ArgumentParser(description='List jobs from the jobs registry.')
    parser.add_argument('--registry', type=str, default='../jobs/jobs_registry.json',
                        help='Path to jobs_registry.json')
    parser.add_argument('--last', type=int, default=None,
                        help='Show only the last N jobs (most recent first)')
    parser.add_argument('--status', type=str, choices=['success', 'failed', 'running'],
                        default=None, help='Filter by status')
    args = parser.parse_args()

    path = os.path.abspath(args.registry)
    jobs = load_registry(path)
    if not jobs:
        print(f"No jobs in {path}")
        return

    # Sort by start_time_iso descending (newest first)
    jobs = sorted(jobs, key=lambda j: j.get('start_time_iso') or '', reverse=True)

    if args.status:
        jobs = [j for j in jobs if j.get('status') == args.status]
    if args.last is not None:
        jobs = jobs[: args.last]

    # Table header
    print(f"{'job_id':<20} {'start_time':<28} {'duration_sec':>12} {'status':<10} output_dir / summary")
    print("-" * 100)
    for j in jobs:
        job_id = (j.get('job_id') or '')[:18]
        start = (j.get('start_time_iso') or '')[:26]
        dur = j.get('duration_sec')
        dur_s = f"{dur:.0f}" if dur is not None else "-"
        status = (j.get('status') or '')[:8]
        out_dir = (j.get('output_dir') or '')[:40]
        summary = j.get('summary')
        if isinstance(summary, dict):
            if summary.get('calibration'):
                one = f"calibration suggested_n_epochs_8h={summary.get('suggested_n_epochs_8h')}"
            else:
                one = f"recon={summary.get('final_recon_loss', 0):.4f} monotone={summary.get('monotone_D_eff')}"
        else:
            one = str(summary)[:30] if summary else ''
        print(f"{job_id:<20} {start:<28} {dur_s:>12} {status:<10} {out_dir} | {one}")
    print(f"\nTotal shown: {len(jobs)}")


if __name__ == '__main__':
    main()
