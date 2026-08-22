"""Recording-level diagnostics for the QiandaoEar22 clip directory.

Run this BEFORE trusting any accuracy number from this repo. It answers the one
question that bounds every result: how many *independent source recordings* back
each class. Clip counts are misleading, because a single continuous recording is
segmented into hundreds of 3-second clips that share the same background noise,
the same hydrophone position, and the same sea state. Two clips from one
recording are not two independent observations.

The clips in this repo are already segmented upstream (outside this repository),
so the only handle on "which recording did this clip come from" is the timestamp
prefix in the filename. This script also checks that the handle is trustworthy.

Usage:
    python dataset_report.py --dataset-dir /path/to/clips [--dataset-dir ...]
    python dataset_report.py --dataset-dir /path/to/clips --json report.json

Requires numpy (via load_data); does not require librosa - no audio is decoded.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from load_data import CLASSES, AudioRecord, assign_sessions, discover_audio_records


_SESSION_OF: dict[str, str] = {}


def _recording_id(record: AudioRecord) -> str:
    """The grouping key used by the split: the continuous recording session.

    Not the timestamp. Timestamps are ~5-minute slices of a continuous session,
    so consecutive timestamps hold audio recorded seconds apart.
    """
    return _SESSION_OF[record.label.timestamp_raw]


def summarize(records: list[AudioRecord]) -> dict[str, object]:
    """Compute clip counts and, more importantly, session counts per class."""
    _SESSION_OF.clear()
    _SESSION_OF.update(assign_sessions(records))
    by_recording: dict[str, list[AudioRecord]] = defaultdict(list)
    for record in records:
        by_recording[_recording_id(record)].append(record)

    class_to_idx = {name: index for index, name in enumerate(CLASSES)}
    uuv_index = class_to_idx["UUV"]

    # For each class: how many clips carry it, and how many distinct recordings.
    clip_counts: Counter[str] = Counter()
    recording_counts: Counter[str] = Counter()
    for name, index in class_to_idx.items():
        clip_counts[name] = sum(int(r.labels_multi[index]) for r in records)
        recording_counts[name] = sum(
            1
            for group in by_recording.values()
            if any(r.labels_multi[index] for r in group)
        )

    # The two audibility variants the notebooks train on separately.
    for variant, attribute in (("UUV-M", "uuv_middle"), ("UUV-W", "uuv_weak")):
        clip_counts[variant] = sum(1 for r in records if getattr(r, attribute))
        recording_counts[variant] = sum(
            1
            for group in by_recording.values()
            if any(getattr(r, attribute) for r in group)
        )

    uuv_recordings = sorted(
        recording_id
        for recording_id, group in by_recording.items()
        if any(r.labels_multi[uuv_index] for r in group)
    )

    # A recording whose clips are not all UUV or all non-UUV means the timestamp
    # groups more than one acoustic situation, so the group key is coarser than
    # intended. That is safe for leakage but costs training data.
    mixed_recordings = [
        recording_id
        for recording_id, group in by_recording.items()
        if 0 < sum(int(r.labels_multi[uuv_index]) for r in group) < len(group)
    ]

    # Duplicate (timestamp, part) pairs mean the same audio segment appears more
    # than once, which inflates counts and can duplicate a clip across the label
    # variants. Grouping by timestamp contains the damage, but it is worth knowing.
    duplicate_parts = 0
    for group in by_recording.values():
        parts = Counter(
            r.label.recording_number
            for r in group
            if r.label.recording_number is not None
        )
        duplicate_parts += sum(count - 1 for count in parts.values() if count > 1)

    sizes = sorted(len(group) for group in by_recording.values())
    return {
        "clip_count": len(records),
        "recording_count": len(by_recording),
        "timestamp_group_count": len({r.label.timestamp_raw for r in records}),
        "clips_per_recording_min": sizes[0],
        "clips_per_recording_median": sizes[len(sizes) // 2],
        "clips_per_recording_max": sizes[-1],
        "duplicate_part_numbers": duplicate_parts,
        "uuv_recording_ids": uuv_recordings,
        "uuv_mixed_recording_ids": sorted(mixed_recordings),
        "clip_counts": dict(clip_counts),
        "recording_counts": dict(recording_counts),
    }


def print_report(summary: dict[str, object]) -> None:
    print(f"Clips:                 {summary['clip_count']}")
    print(f"Timestamp groups:      {summary['timestamp_group_count']}  (~5 min each, NOT independent)")
    print(f"Continuous sessions:   {summary['recording_count']}  <- the real grouping unit")
    print(
        "Clips per recording:   "
        f"min={summary['clips_per_recording_min']} "
        f"median={summary['clips_per_recording_median']} "
        f"max={summary['clips_per_recording_max']}"
    )
    print(f"Duplicate (ts, part):  {summary['duplicate_part_numbers']}")
    print()

    clip_counts: dict[str, int] = summary["clip_counts"]  # type: ignore[assignment]
    recording_counts: dict[str, int] = summary["recording_counts"]  # type: ignore[assignment]
    print(f"{'class':<20}{'clips':>8}{'recordings':>12}   note")
    for name in list(CLASSES) + ["UUV-M", "UUV-W"]:
        clips = clip_counts.get(name, 0)
        recordings = recording_counts.get(name, 0)
        if clips == 0:
            note = "DEAD - never occurs, drop from CLASSES"
        elif recordings < 5:
            note = "TOO FEW RECORDINGS - cannot be evaluated honestly"
        elif recordings < 10:
            note = "very few recordings - expect huge fold variance"
        else:
            note = ""
        print(f"{name:<20}{clips:>8}{recordings:>12}   {note}")

    print()
    uuv_recordings: list[str] = summary["uuv_recording_ids"]  # type: ignore[assignment]
    print(f"UUV-positive recordings ({len(uuv_recordings)}): {uuv_recordings}")
    mixed: list[str] = summary["uuv_mixed_recording_ids"]  # type: ignore[assignment]
    if mixed:
        print(
            f"Recordings with both UUV and non-UUV clips ({len(mixed)}): {mixed}\n"
            "  These timestamps cover more than one acoustic situation."
        )

    print()
    if summary["duplicate_part_numbers"]:
        print(
            f"{summary['duplicate_part_numbers']} of {summary['clip_count']} clips share a "
            "(timestamp, part) pair with\n"
            "  another clip. Either the part number is not a per-clip index, or the same\n"
            "  audio segment is stored more than once. Re-run with --show-duplicates to\n"
            "  see filenames and decide which.\n"
        )

    print("How to read this: the 'recordings' column is the effective sample size")
    print("for generalisation. A class with 1900 clips from 10 recordings gives a")
    print("model 10 independent examples, not 1900. Confidence intervals on any")
    print("held-out score should be computed over recordings, never over clips.")


def show_duplicates(records: list[AudioRecord], limit: int = 5) -> None:
    """Print filenames that share a (timestamp, part) pair.

    If the names differ only in their target list, the part number is a label
    index rather than a clip index and nothing is duplicated. If the names are
    identical apart from a directory, the same audio really is stored twice and
    the effective dataset size is smaller than the clip count suggests.
    """
    groups: dict[tuple[str, object], list[AudioRecord]] = defaultdict(list)
    for record in records:
        if record.label.recording_number is not None:
            groups[(record.label.timestamp_raw, record.label.recording_number)].append(record)

    shown = 0
    for key, group in sorted(groups.items()):
        if len(group) < 2 or shown >= limit:
            continue
        print(f"\ntimestamp {key[0]}, part {key[1]} -> {len(group)} files:")
        for record in group[:6]:
            print(f"    {record.path.name}")
        shown += 1


def main(
    dataset_dirs: Iterable[str | Path],
    json_path: str | Path | None,
    show_duplicate_examples: bool = False,
) -> None:
    records = discover_audio_records(dataset_dirs)
    summary = summarize(records)
    print_report(summary)
    if show_duplicate_examples:
        show_duplicates(records)
    if json_path is not None:
        Path(json_path).write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"\nWrote {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", action="append", required=True)
    parser.add_argument("--json", default=None, help="Also write the summary as JSON.")
    parser.add_argument(
        "--show-duplicates",
        action="store_true",
        help="Print filenames sharing a (timestamp, part) pair.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.dataset_dir, args.json, show_duplicate_examples=args.show_duplicates)
