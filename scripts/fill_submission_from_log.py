#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def extract(pattern: str, text: str, cast=None):
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None
    value = m.group(1)
    return cast(value) if cast else value


def parse_log(log_text: str) -> dict:
    out: dict = {}

    val_loss = extract(
        r"final_int8_zlib_roundtrip_exact val_loss:([0-9.]+)",
        log_text,
        float,
    )
    val_bpb = extract(
        r"final_int8_zlib_roundtrip_exact val_bpb:([0-9.]+)",
        log_text,
        float,
    )
    if val_loss is None:
        val_loss = extract(r"final_int8_zlib_roundtrip val_loss:([0-9.]+)", log_text, float)
    if val_bpb is None:
        val_bpb = extract(r"final_int8_zlib_roundtrip val_bpb:([0-9.]+)", log_text, float)

    step_stop = extract(r"stopping_early:.* step:([0-9]+)/[0-9]+", log_text, int)
    wallclock_ms = extract(r"stopping_early:.* train_time:([0-9]+)ms", log_text, int)
    bytes_total = extract(r"Total submission size int8\+zlib:\s*([0-9]+)\s*bytes", log_text, int)
    bytes_model_int8_zlib = extract(
        r"Serialized model int8\+zlib:\s*([0-9]+)\s*bytes",
        log_text,
        int,
    )
    bytes_code = extract(r"Code size:\s*([0-9]+)\s*bytes", log_text, int)

    out["val_loss"] = val_loss
    out["val_bpb"] = val_bpb
    out["step_stop"] = step_stop
    out["wallclock_seconds"] = None if wallclock_ms is None else int(round(wallclock_ms / 1000.0))
    out["bytes_total"] = bytes_total
    out["bytes_model_int8_zlib"] = bytes_model_int8_zlib
    out["bytes_code"] = bytes_code
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True, help="Path to submission.json")
    ap.add_argument("--log", required=True, help="Path to train log")
    ap.add_argument("--author")
    ap.add_argument("--github-id")
    ap.add_argument("--name")
    ap.add_argument("--blurb")
    ap.add_argument("--gpu")
    ap.add_argument("--track")
    args = ap.parse_args()

    submission_path = Path(args.submission)
    log_path = Path(args.log)

    submission = {}
    if submission_path.exists():
        submission = json.loads(submission_path.read_text(encoding="utf-8"))

    parsed = parse_log(log_path.read_text(encoding="utf-8", errors="ignore"))
    submission.update({k: v for k, v in parsed.items() if v is not None})

    if args.author is not None:
        submission["author"] = args.author
    if args.github_id is not None:
        submission["github_id"] = args.github_id
    if args.name is not None:
        submission["name"] = args.name
    if args.blurb is not None:
        submission["blurb"] = args.blurb
    if args.gpu is not None:
        submission["gpu"] = args.gpu
    if args.track is not None:
        submission["track"] = args.track

    submission["date"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")

    print("Updated:", submission_path)
    for key in [
        "val_bpb",
        "val_loss",
        "step_stop",
        "wallclock_seconds",
        "bytes_total",
        "bytes_model_int8_zlib",
        "bytes_code",
    ]:
        print(f"{key}: {submission.get(key)}")


if __name__ == "__main__":
    main()

