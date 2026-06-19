"""Download a small OpenNeuro subset for external sleep-dataset smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import openneuro


def build_include_patterns(subjects: list[str] | None) -> list[str]:
    include = [
        "dataset_description.json",
        "participants.*",
        "README*",
        "CHANGES*",
    ]
    if not subjects:
        include.append("sub-*/eeg/*")
        return include

    for subject in subjects:
        subject = subject.removeprefix("sub-")
        include.append(f"sub-{subject}/eeg/*")
    return include


def main() -> None:
    parser = argparse.ArgumentParser(description="Download an OpenNeuro EEG subset")
    parser.add_argument("--dataset", required=True, help="OpenNeuro dataset id, e.g. ds006695")
    parser.add_argument("--target-dir", required=True)
    parser.add_argument(
        "--subject",
        action="append",
        default=None,
        help="Subject id to include, e.g. 01. Repeat for multiple subjects. Omit for all EEG files.",
    )
    parser.add_argument("--tag", default=None, help="Optional OpenNeuro version tag")
    parser.add_argument("--max-concurrent-downloads", type=int, default=2)
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    include = build_include_patterns(args.subject)
    print(f"Downloading {args.dataset} to {target_dir.resolve()}")
    print("Include patterns:")
    for pattern in include:
        print(f"  {pattern}")

    openneuro.download(
        dataset=args.dataset,
        tag=args.tag,
        target_dir=target_dir,
        include=include,
        max_concurrent_downloads=args.max_concurrent_downloads,
    )


if __name__ == "__main__":
    main()
