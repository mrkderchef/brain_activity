"""Download a small Sleep-EDF Expanded subset from PhysioNet."""

from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path


PHYSIONET_SLEEP_EDF_URL = "https://physionet.org/files/sleep-edfx/1.0.0"
DEFAULT_RECORDS = ["SC4001E0", "SC4011E0"]


def read_records(base_url: str) -> list[str]:
    records_url = f"{base_url.rstrip('/')}/RECORDS"
    with urllib.request.urlopen(records_url, timeout=60) as response:
        text = response.read().decode("utf-8")
    return [item.strip() for item in text.split() if item.strip()]


def read_directory_files(base_url: str, folder: str) -> list[str]:
    index_url = f"{base_url.rstrip('/')}/{folder.strip('/')}/"
    with urllib.request.urlopen(index_url, timeout=60) as response:
        text = response.read().decode("utf-8")
    filenames = set(re.findall(r">([^<>]+?\.edf)<", text, flags=re.IGNORECASE))
    filenames.update(re.findall(r'href="([^"]+?\.edf)"', text, flags=re.IGNORECASE))
    return sorted(filename for filename in filenames if not filename.startswith("../"))


def build_pairs(records: list[str]) -> dict[str, dict[str, str]]:
    pairs: dict[str, dict[str, str]] = {}
    for record in records:
        name = Path(record).name
        if name.endswith("-PSG.edf"):
            key = name[:6]
            pairs.setdefault(key, {})["psg"] = record
        elif name.endswith("-Hypnogram.edf"):
            key = name[:6]
            pairs.setdefault(key, {})["hypnogram"] = record
    return {key: value for key, value in pairs.items() if {"psg", "hypnogram"} <= set(value)}


def add_hypnograms_from_directory_indexes(base_url: str, pairs: dict[str, dict[str, str]]) -> None:
    folders = sorted({str(Path(pair["psg"]).parent).replace("\\", "/") for pair in pairs.values() if "psg" in pair})
    for folder in folders:
        for filename in read_directory_files(base_url, folder):
            if not filename.endswith("-Hypnogram.edf"):
                continue
            key = filename[:6].upper()
            if key in pairs:
                pairs[key]["hypnogram"] = f"{folder}/{filename}"


def select_pairs(
    pairs: dict[str, dict[str, str]],
    requested_records: list[str] | None,
    max_records: int | None,
) -> list[tuple[str, dict[str, str]]]:
    if requested_records:
        requested = [record.upper().removesuffix("-PSG.EDF") for record in requested_records]
        selected = []
        for record in requested:
            key = record[:6]
            if key not in pairs:
                raise KeyError(f"No complete Sleep-EDF PSG/Hypnogram pair found for {record!r}")
            selected.append((key, pairs[key]))
        return selected

    if max_records is not None:
        selected = sorted(pairs.items())[:max_records]
        return selected

    selected = [(record[:6], pairs[record[:6]]) for record in DEFAULT_RECORDS if record[:6] in pairs]
    return selected


def download_file(url: str, target_path: Path, force: bool = False) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not force:
        print(f"Skipping existing {target_path}")
        return
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, target_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small Sleep-EDF Expanded subset")
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--base-url", default=PHYSIONET_SLEEP_EDF_URL)
    parser.add_argument(
        "--record",
        action="append",
        default=None,
        help="Sleep-EDF record id such as SC4001E0. Repeat for multiple records.",
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = read_records(args.base_url)
    psg_pairs = {
        Path(record).name[:6].upper(): {"psg": record}
        for record in records
        if Path(record).name.endswith("-PSG.edf")
    }
    add_hypnograms_from_directory_indexes(args.base_url, psg_pairs)
    pairs = {key: value for key, value in psg_pairs.items() if {"psg", "hypnogram"} <= set(value)}
    selected = select_pairs(pairs, args.record, args.max_records)
    if not selected:
        raise RuntimeError("No Sleep-EDF record pairs selected")

    target_dir = Path(args.target_dir)
    print("Selected Sleep-EDF pairs:")
    for key, pair in selected:
        print(f"  {key}:")
        for kind in ["psg", "hypnogram"]:
            source_path = pair[kind]
            print(f"    {source_path}")
            if not args.dry_run:
                download_file(
                    url=f"{args.base_url.rstrip('/')}/{source_path}",
                    target_path=target_dir / source_path,
                    force=args.force,
                )


if __name__ == "__main__":
    main()
