"""
Proteus — Replay Buffer Builder
Run this after training on a domain to append its eval samples
into the shared replay buffer used by the Replay condition.

Usage:
  python build_replay_buffer.py --domain medical
  python build_replay_buffer.py --domain legal
  python build_replay_buffer.py --domain code
  python build_replay_buffer.py --domain multilingual

Typical sequence for Replay condition:
  python train.py   --domain medical    --condition replay
  python build_replay_buffer.py         --domain medical
  python train.py   --domain legal      --condition replay --replay_buffer data/replay_buffer.jsonl
  python build_replay_buffer.py         --domain legal
  python train.py   --domain code       --condition replay --replay_buffer data/replay_buffer.jsonl
  python build_replay_buffer.py         --domain code
  python train.py   --domain multilingual --condition replay --replay_buffer data/replay_buffer.jsonl
"""

import argparse
import json
import random
from pathlib import Path

DATA_DIR   = Path("data")
BUFFER     = DATA_DIR / "replay_buffer.jsonl"
SAMPLES_PER_DOMAIN = 500   # how many eval samples to pull into the buffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True,
                        choices=["medical", "legal", "code", "multilingual"])
    parser.add_argument("--n_samples", type=int, default=SAMPLES_PER_DOMAIN,
                        help="Number of eval samples to add to the buffer.")
    args = parser.parse_args()

    eval_path = DATA_DIR / args.domain / "eval.jsonl"
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_path}. Run pipeline.py first.")

    # Load eval records
    records = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    # Sample (eval set is exactly 500 so this is usually all of them)
    sampled = random.sample(records, min(args.n_samples, len(records)))

    # Check what's already in the buffer to avoid exact duplicates
    existing = set()
    if BUFFER.exists():
        with open(BUFFER, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                existing.add(r["instruction"][:80] + r["input"][:80])

    new_records = [
        r for r in sampled
        if (r["instruction"][:80] + r["input"][:80]) not in existing
    ]

    # Append to buffer
    DATA_DIR.mkdir(exist_ok=True)
    with open(BUFFER, "a", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = sum(1 for _ in open(BUFFER, encoding="utf-8"))
    print(f"[replay] Added {len(new_records)} records from {args.domain}.")
    print(f"[replay] Buffer total: {total} records -> {BUFFER}")


if __name__ == "__main__":
    main()
