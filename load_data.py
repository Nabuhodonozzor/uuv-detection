from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import librosa
import numpy as np


CLASSES = [
    "ArtificialSignals", "BigPassengerShip", "Cargo", "FishBoat", "GreenCity",
    "KaiYan", "KaiYuan", "MotorBoat", "No7", "No5", "PoliceBoat", "QianDao",
    "SpeedBoat", "TheEarl", "TheKnight", "UUV", "Unknown", "WorkShip", "Helicopter", "CivilianBoats", "Car"
]

TARGET_MODE_MAP = {"S": "single", "M": "multiple"}
DISTANCE_MAP = {"N": "near", "M": "medium", "F": "far"}
AUDIBILITY_MAP = {"S": "strong", "M": "middle", "W": "weak"}
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}
FEATURE_FORMAT_VERSION = "2"


@dataclass
class TargetInfo:
    name: str
    distance_code: str
    distance: str
    audibility_code: str
    audibility: str


@dataclass
class AudioLabel:
    filename: str
    timestamp_raw: str
    timestamp: str
    target_mode_code: str
    target_mode: str
    targets: List[TargetInfo]
    label: Optional[int]
    recording_number: Optional[int]


@dataclass
class AudioRecord:
    path: Path
    sample_id: str
    label: AudioLabel
    labels_multi: np.ndarray
    uuv_middle: bool
    uuv_weak: bool


def parse_audio_filename(filename: str) -> AudioLabel:
    """Parse the QiandaoEar22 metadata encoded in an audio filename."""
    path = Path(filename)
    stem = path.stem
    if "_label__" not in stem:
        raise ValueError(f"Filename does not contain '_label__': {filename}")

    metadata_part, label_part = stem.split("_label__", maxsplit=1)
    label_tokens = label_part.split("__")
    label = int(label_tokens[0]) if label_tokens and label_tokens[0] else None
    recording_number = (
        int(label_tokens[1]) if len(label_tokens) >= 2 and label_tokens[1] else None
    )

    parts = metadata_part.split("_")
    if len(parts) < 3:
        raise ValueError(f"Filename metadata is too short: {filename}")

    timestamp_raw, target_mode_code = parts[:2]
    if target_mode_code not in TARGET_MODE_MAP:
        raise ValueError(
            f"Unknown target mode code '{target_mode_code}' in {filename}"
        )

    timestamp = datetime.strptime(timestamp_raw, "%Y%m%d%H%M%S").isoformat()
    targets: List[TargetInfo] = []
    for raw_target in "_".join(parts[2:]).split("&"):
        target_parts = raw_target.split("_")

        # Some files append a numeric metadata field after the target description.
        while len(target_parts) > 1 and target_parts[-1].isdigit():
            target_parts.pop()

        if (
            len(target_parts) >= 3
            and target_parts[-2] in DISTANCE_MAP
            and target_parts[-1] in AUDIBILITY_MAP
        ):
            name = "_".join(target_parts[:-2])
            distance_code = target_parts[-2]
            audibility_code = target_parts[-1]
            distance = DISTANCE_MAP[distance_code]
            audibility = AUDIBILITY_MAP[audibility_code]
        else:
            name = "_".join(target_parts)
            distance_code = ""
            audibility_code = ""
            distance = "unknown"
            audibility = "unknown"

        targets.append(
            TargetInfo(
                name=name,
                distance_code=distance_code,
                distance=distance,
                audibility_code=audibility_code,
                audibility=audibility,
            )
        )

    return AudioLabel(
        filename=filename,
        timestamp_raw=timestamp_raw,
        timestamp=timestamp,
        target_mode_code=target_mode_code,
        target_mode=TARGET_MODE_MAP[target_mode_code],
        targets=targets,
        label=label,
        recording_number=recording_number,
    )


def extract_mfcc(
    audio_path: str | Path,
    sample_rate: Optional[int] = None,
    n_mfcc: int = 10,
    n_fft: int = 2048,
    hop_length: int = 256,
    n_mels: int = 256,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    omit_zero_order: bool = True,
) -> np.ndarray:
    """Extract MFCCs with shape ``(time_frames, n_mfcc)``."""
    y, sr = librosa.load(audio_path, sr=sample_rate)
    effective_fmax = sr / 2 if fmax is None else fmax
    requested_n_mfcc = n_mfcc + 1 if omit_zero_order else n_mfcc
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=requested_n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=effective_fmax,
    )
    if omit_zero_order:
        mfcc = mfcc[1:, :]
    return np.asarray(mfcc.T, dtype=np.float32)


def discover_audio_records(dataset_dirs: Iterable[str | Path]) -> list[AudioRecord]:
    """Load labels before feature extraction and assign stable sample IDs."""
    class_to_idx = {class_name: idx for idx, class_name in enumerate(CLASSES)}
    records: list[AudioRecord] = []
    seen_ids: set[str] = set()

    for source_index, dataset_dir_value in enumerate(dataset_dirs):
        dataset_dir = Path(dataset_dir_value)
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

        audio_paths = sorted(
            path
            for path in dataset_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )
        for audio_path in audio_paths:
            relative_path = audio_path.relative_to(dataset_dir).as_posix()
            sample_key = f"{source_index}:{relative_path}"
            sample_id = hashlib.sha256(sample_key.encode("utf-8")).hexdigest()[:20]
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample ID generated for {audio_path}")
            seen_ids.add(sample_id)

            label = parse_audio_filename(audio_path.name)
            labels_multi = np.zeros(len(CLASSES), dtype=np.int8)
            unknown_targets = sorted(
                {target.name for target in label.targets if target.name not in class_to_idx}
            )
            if unknown_targets:
                raise ValueError(
                    f"Unknown target classes {unknown_targets} in {audio_path.name}"
                )
            for target in label.targets:
                labels_multi[class_to_idx[target.name]] = 1

            uuv_audibility = {
                target.audibility_code for target in label.targets if target.name == "UUV"
            }
            records.append(
                AudioRecord(
                    path=audio_path,
                    sample_id=sample_id,
                    label=label,
                    labels_multi=labels_multi,
                    uuv_middle="M" in uuv_audibility,
                    uuv_weak="W" in uuv_audibility,
                )
            )

    if not records:
        raise ValueError("No supported audio files were found.")
    return records


def _group_statistics(
    records: list[AudioRecord],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    groups: dict[str, list[AudioRecord]] = {}
    for record in records:
        groups.setdefault(record.label.timestamp_raw, []).append(record)

    timestamps = sorted(groups)
    group_sizes = np.asarray([len(groups[timestamp]) for timestamp in timestamps])
    label_counts = []
    label_group_support = []
    duplicate_timestamp_count = 0
    duplicate_part_count = 0
    duplicate_examples: list[dict[str, object]] = []
    for timestamp in timestamps:
        group_records = groups[timestamp]
        counts = np.sum([record.labels_multi for record in group_records], axis=0)
        counts = np.concatenate(
            [
                counts,
                [sum(record.uuv_middle for record in group_records)],
                [sum(record.uuv_weak for record in group_records)],
            ]
        )
        label_counts.append(counts)
        label_group_support.append(counts > 0)

        records_by_part: dict[int, list[AudioRecord]] = {}
        for record in group_records:
            if record.label.recording_number is not None:
                records_by_part.setdefault(record.label.recording_number, []).append(record)
        duplicate_parts = {
            part_number: part_records
            for part_number, part_records in records_by_part.items()
            if len(part_records) > 1
        }
        if duplicate_parts:
            duplicate_timestamp_count += 1
            duplicate_part_count += sum(
                len(part_records) - 1 for part_records in duplicate_parts.values()
            )
            for part_number, part_records in duplicate_parts.items():
                if len(duplicate_examples) >= 5:
                    break
                duplicate_examples.append(
                    {
                        "timestamp": timestamp,
                        "recording_number": part_number,
                        "filenames": [record.path.name for record in part_records[:5]],
                    }
                )

    group_label_counts = np.asarray(label_counts, dtype=np.float64)
    group_support = np.sum(label_group_support, axis=0)
    diagnostics = {
        "audio_file_count": len(records),
        "timestamp_group_count": len(timestamps),
        "timestamps_with_duplicate_part_numbers": duplicate_timestamp_count,
        "duplicate_part_number_occurrences": duplicate_part_count,
        "duplicate_part_number_examples": duplicate_examples,
    }
    if duplicate_part_count:
        print(
            "Warning: duplicate part numbers were found within "
            f"{duplicate_timestamp_count} timestamp groups "
            f"({duplicate_part_count} additional occurrences). "
            "All files will be preserved and grouped by timestamp."
        )
        for example in duplicate_examples:
            print(
                f"  {example['timestamp']} part {example['recording_number']}: "
                f"{example['filenames']}"
            )
    return timestamps, group_sizes, group_label_counts, group_support, diagnostics


def _assignment_score(
    sample_counts: np.ndarray,
    label_counts: np.ndarray,
    target_samples: np.ndarray,
    target_labels: np.ndarray,
    enforce_labels: np.ndarray,
) -> float:
    sample_error = np.mean(
        np.square((sample_counts - target_samples) / np.maximum(target_samples, 1.0))
    )
    valid_labels = np.sum(target_labels, axis=0) > 0
    label_error = np.mean(
        np.square(
            (label_counts[:, valid_labels] - target_labels[:, valid_labels])
            / np.maximum(target_labels[:, valid_labels], 1.0)
        )
    )
    missing = np.sum((label_counts[:, enforce_labels] == 0))
    return 3.0 * sample_error + label_error + 100.0 * missing


def assign_test_and_folds(
    records: list[AudioRecord],
    n_splits: int = 5,
    test_size: float = 0.2,
    seed: int = 42,
    candidate_count: int = 2048,
) -> tuple[dict[str, str], str, dict[str, object]]:
    """Assign complete timestamp groups to one test bucket or one CV fold."""
    if n_splits != 5:
        raise ValueError("QiandaoEar22 experiments use exactly 5 CV folds.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    timestamps, group_sizes, group_labels, group_support, diagnostics = (
        _group_statistics(records)
    )
    bucket_count = n_splits + 1
    if len(timestamps) < bucket_count:
        raise ValueError(
            f"At least {bucket_count} timestamp groups are required, found {len(timestamps)}."
        )

    fractions = np.asarray(
        [test_size] + [(1.0 - test_size) / n_splits] * n_splits,
        dtype=np.float64,
    )
    target_samples = fractions * np.sum(group_sizes)
    target_labels = fractions[:, np.newaxis] * np.sum(group_labels, axis=0)
    enforce_labels = group_support >= bucket_count
    rng = np.random.default_rng(seed)
    best_assignment: Optional[np.ndarray] = None
    best_score = np.inf

    for _ in range(candidate_count):
        order = rng.permutation(len(timestamps))
        sample_counts = np.zeros(bucket_count, dtype=np.float64)
        label_counts = np.zeros((bucket_count, group_labels.shape[1]), dtype=np.float64)
        assignment = np.full(len(timestamps), -1, dtype=np.int8)

        for group_index in order:
            bucket_scores = []
            for bucket_index in range(bucket_count):
                candidate_samples = sample_counts.copy()
                candidate_labels = label_counts.copy()
                candidate_samples[bucket_index] += group_sizes[group_index]
                candidate_labels[bucket_index] += group_labels[group_index]
                bucket_scores.append(
                    _assignment_score(
                        candidate_samples,
                        candidate_labels,
                        target_samples,
                        target_labels,
                        np.zeros_like(enforce_labels, dtype=bool),
                    )
                )
            minimum = min(bucket_scores)
            choices = np.flatnonzero(np.isclose(bucket_scores, minimum))
            selected_bucket = int(rng.choice(choices))
            assignment[group_index] = selected_bucket
            sample_counts[selected_bucket] += group_sizes[group_index]
            label_counts[selected_bucket] += group_labels[group_index]

        score = _assignment_score(
            sample_counts,
            label_counts,
            target_samples,
            target_labels,
            enforce_labels,
        )
        if score < best_score:
            best_score = score
            best_assignment = assignment.copy()

    if best_assignment is None:
        raise RuntimeError("Could not create grouped test and fold assignments.")

    actual_samples = np.zeros(bucket_count, dtype=np.float64)
    actual_labels = np.zeros((bucket_count, group_labels.shape[1]), dtype=np.float64)
    for group_index, bucket_index in enumerate(best_assignment):
        actual_samples[bucket_index] += group_sizes[group_index]
        actual_labels[bucket_index] += group_labels[group_index]

    if np.any(actual_samples == 0):
        raise ValueError(f"Grouped assignment produced an empty bucket: {actual_samples}")
    missing_balanced_labels = np.argwhere(
        (actual_labels[:, enforce_labels] == 0)
    )
    if len(missing_balanced_labels):
        raise ValueError(
            "Could not place every sufficiently represented class in every test/CV bucket."
        )

    total_samples = float(np.sum(group_sizes))
    actual_fractions = actual_samples / total_samples
    maximum_group_fraction = float(np.max(group_sizes) / total_samples)
    tolerance = max(0.01, maximum_group_fraction)
    if np.any(np.abs(actual_fractions - fractions) > tolerance):
        raise ValueError(
            "Grouped split is outside the allowed sample-ratio tolerance: "
            f"actual={actual_fractions.tolist()}, target={fractions.tolist()}, "
            f"tolerance={tolerance:.4f}"
        )

    uuv_index = CLASSES.index("UUV")
    negative_counts = actual_samples - actual_labels[:, uuv_index]
    for variant_name, label_index in {
        "M": len(CLASSES),
        "W": len(CLASSES) + 1,
    }.items():
        if np.any(actual_labels[:, label_index] == 0) or np.any(negative_counts == 0):
            raise ValueError(
                f"Every test/CV bucket must contain positive and negative samples for UUV-{variant_name}. "
                f"Positive timestamp groups: {int(group_support[label_index])}; "
                f"positive samples by [test, fold_0..fold_{n_splits - 1}]: "
                f"{actual_labels[:, label_index].astype(int).tolist()}; "
                f"negative samples: {negative_counts.astype(int).tolist()}."
            )

    bucket_names = ["test"] + [f"fold_{fold}" for fold in range(n_splits)]
    timestamp_assignment = {
        timestamp: bucket_names[int(best_assignment[index])]
        for index, timestamp in enumerate(timestamps)
    }
    sample_assignment = {
        record.sample_id: timestamp_assignment[record.label.timestamp_raw]
        for record in records
    }

    split_rows = sorted(sample_assignment.items())
    split_id = hashlib.sha256(
        "\n".join(f"{sample_id}:{bucket}" for sample_id, bucket in split_rows).encode(
            "utf-8"
        )
    ).hexdigest()

    counts = {
        bucket: sum(value == bucket for value in sample_assignment.values())
        for bucket in bucket_names
    }
    print(f"Timestamp groups: {len(timestamps)}")
    print(f"Samples per bucket: {counts}")
    print(f"Split ID: {split_id}")
    return sample_assignment, split_id, diagnostics


def _stack_features(features: list[np.ndarray], split_name: str) -> np.ndarray:
    try:
        return np.stack(features).astype(np.float32, copy=False)
    except ValueError as exc:
        shapes = sorted({feature.shape for feature in features})
        raise ValueError(
            f"MFCC samples in {split_name} have inconsistent shapes: {shapes}"
        ) from exc


def create_mfcc_archive(
    records: list[AudioRecord],
    assignment: dict[str, str],
    split_id: str,
    output_path: str | Path,
    n_mfcc: int,
    n_splits: int = 5,
    dataset_diagnostics: Optional[dict[str, object]] = None,
) -> Path:
    """Extract one MFCC configuration and save its fixed test/CV assignment."""
    cv_records = [record for record in records if assignment[record.sample_id] != "test"]
    test_records = [record for record in records if assignment[record.sample_id] == "test"]
    cv_features = []
    test_features = []

    for index, record in enumerate(records, start=1):
        print(f"MFCC-{n_mfcc}: {index}/{len(records)} {record.path.name}")
        feature = extract_mfcc(record.path, n_mfcc=n_mfcc)
        if assignment[record.sample_id] == "test":
            test_features.append(feature)
        else:
            cv_features.append(feature)

    uuv_index = CLASSES.index("UUV")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.npz")
    feature_config = {
        "sample_rate": None,
        "n_mfcc": n_mfcc,
        "n_fft": 2048,
        "hop_length": 256,
        "n_mels": 256,
        "fmin": 0.0,
        "fmax": None,
        "omit_zero_order": True,
    }
    total_samples = len(records)
    test_fraction = len(test_records) / total_samples
    fold_fractions = np.asarray(
        [
            sum(assignment[record.sample_id] == f"fold_{fold}" for record in records)
            / total_samples
            for fold in range(n_splits)
        ],
        dtype=np.float32,
    )
    split_ratios = np.column_stack(
        [1.0 - test_fraction - fold_fractions, fold_fractions, np.full(n_splits, test_fraction)]
    ).astype(np.float32)

    np.savez_compressed(
        temporary_path,
        cv_data=_stack_features(cv_features, "cv"),
        cv_labels_multi=np.stack([record.labels_multi for record in cv_records]),
        cv_labels_binary=np.asarray(
            [record.labels_multi[uuv_index] for record in cv_records], dtype=np.int8
        ),
        cv_fold_ids=np.asarray(
            [int(assignment[record.sample_id].split("_")[1]) for record in cv_records],
            dtype=np.int8,
        ),
        cv_sample_ids=np.asarray([record.sample_id for record in cv_records]),
        cv_timestamps=np.asarray([record.label.timestamp_raw for record in cv_records]),
        cv_uuv_middle=np.asarray([record.uuv_middle for record in cv_records]),
        cv_uuv_weak=np.asarray([record.uuv_weak for record in cv_records]),
        test_data=_stack_features(test_features, "test"),
        test_labels_multi=np.stack([record.labels_multi for record in test_records]),
        test_labels_binary=np.asarray(
            [record.labels_multi[uuv_index] for record in test_records], dtype=np.int8
        ),
        test_sample_ids=np.asarray([record.sample_id for record in test_records]),
        test_timestamps=np.asarray([record.label.timestamp_raw for record in test_records]),
        test_uuv_middle=np.asarray([record.uuv_middle for record in test_records]),
        test_uuv_weak=np.asarray([record.uuv_weak for record in test_records]),
        classes=np.asarray(CLASSES),
        n_mfcc=np.asarray(n_mfcc, dtype=np.int16),
        n_folds=np.asarray(n_splits, dtype=np.int8),
        split_id=np.asarray(split_id),
        split_ratios=split_ratios,
        group_key=np.asarray("timestamp_raw"),
        dataset_diagnostics_json=np.asarray(
            json.dumps(dataset_diagnostics or {}, sort_keys=True)
        ),
        feature_config_json=np.asarray(json.dumps(feature_config, sort_keys=True)),
        format_version=np.asarray(FEATURE_FORMAT_VERSION),
    )
    temporary_path.replace(output_path)
    print(f"Saved {output_path}")
    return output_path


def build_all_mfcc_archives(
    dataset_dirs: Iterable[str | Path],
    output_dir: str | Path,
    n_splits: int = 5,
    seed: int = 42,
) -> list[Path]:
    records = discover_audio_records(dataset_dirs)
    print(f"Loaded metadata for {len(records)} audio files.")
    assignment, split_id, dataset_diagnostics = assign_test_and_folds(
        records, n_splits=n_splits, test_size=0.2, seed=seed
    )
    output_dir = Path(output_dir)
    return [
        create_mfcc_archive(
            records,
            assignment,
            split_id,
            output_dir / f"mfcc{n_mfcc}.npz",
            n_mfcc=n_mfcc,
            n_splits=n_splits,
            dataset_diagnostics=dataset_diagnostics,
        )
        for n_mfcc in (10, 20, 40)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create MFCC-10/20/40 archives with one grouped test split and shared CV folds."
    )
    parser.add_argument(
        "--dataset-dir",
        action="append",
        required=True,
        help="Audio dataset directory. Repeat this option for every source directory.",
    )
    parser.add_argument("--output-dir", default="mfcc_datasets")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_all_mfcc_archives(
        args.dataset_dir,
        args.output_dir,
        n_splits=args.n_splits,
        seed=args.seed,
    )
