from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import json
import numpy as np
import librosa


# -----------------------------
# Label classes
# -----------------------------

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


# -----------------------------
# Decoding dictionaries
# -----------------------------

TARGET_MODE_MAP = {
    "S": "single",
    "M": "multiple",
}

DISTANCE_MAP = {
    "N": "near",
    "M": "medium",
    "F": "far",
}

AUDIBILITY_MAP = {
    "S": "strong",
    "M": "middle",
    "W": "weak",
}


# -----------------------------
# Filename parser
# -----------------------------

def parse_audio_filename(filename: str) -> AudioLabel:
    """
    Parse filename format:

    timestamp_single/multiple_sources_label__label_id__recording_number

    Example:
    20220624162953_M_UUV_M_M&MotorBoat_M_M&ArtificialSignals_0_label__1__35

    Meaning:
    20220624162953  -> timestamp
    M               -> multiple targets
    UUV_M_M         -> target: UUV, distance M, audibility M
    MotorBoat_M_M   -> target: MotorBoat, distance M, audibility M
    ArtificialSignals_0 -> target: ArtificialSignals, distance/audibility unknown or special
    label__1        -> class label 1
    35              -> recording number
    """

    path = Path(filename)
    stem = path.stem

    # Split label section from main metadata
    if "_label__" not in stem:
        raise ValueError(f"Filename does not contain '_label__': {filename}")

    metadata_part, label_part = stem.split("_label__", maxsplit=1)

    label_tokens = label_part.split("__")

    label = None
    recording_number = None

    if len(label_tokens) >= 1 and label_tokens[0] != "":
        label = int(label_tokens[0])

    if len(label_tokens) >= 2 and label_tokens[1] != "":
        recording_number = int(label_tokens[1])

    # Split main metadata
    parts = metadata_part.split("_")

    if len(parts) < 3:
        raise ValueError(f"Filename metadata is too short: {filename}")

    timestamp_raw = parts[0]
    target_mode_code = parts[1]

    if target_mode_code not in TARGET_MODE_MAP:
        raise ValueError(f"Unknown target mode code '{target_mode_code}' in {filename}")

    timestamp = datetime.strptime(timestamp_raw, "%Y%m%d%H%M%S").isoformat()

    target_mode = TARGET_MODE_MAP[target_mode_code]

    # Everything after timestamp and S/M mode belongs to target description
    target_string = "_".join(parts[2:])

    raw_targets = target_string.split("&")

    targets: List[TargetInfo] = []

    for raw_target in raw_targets:
        target_parts = raw_target.split("_")

        # Normal case: Name_Distance_Audibility
        # Example: UUV_M_M
        if len(target_parts) >= 3:
            name = "_".join(target_parts[:-2])
            distance_code = target_parts[-2]
            audibility_code = target_parts[-1]

            distance = DISTANCE_MAP.get(distance_code, "unknown")
            audibility = AUDIBILITY_MAP.get(audibility_code, "unknown")

        # Special or incomplete case, for example: ArtificialSignals_0
        elif len(target_parts) == 2:
            name = target_parts[0]
            distance_code = target_parts[1]
            audibility_code = ""

            distance = DISTANCE_MAP.get(distance_code, "unknown")
            audibility = "unknown"

        else:
            name = raw_target
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
        target_mode=target_mode,
        targets=targets,
        label=label,
        recording_number=recording_number,
    )


# -----------------------------
# MFCC extraction
# -----------------------------

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
    """
    Extract variable-length MFCCs.

    If omit_zero_order=True, MFCC coefficient 0 is removed.

    Returns:
        np.ndarray with shape (time_frames, n_mfcc)

    Important:
        When omit_zero_order=True, this function internally computes
        n_mfcc + 1 coefficients, then removes coefficient 0, so the final
        output still has n_mfcc columns.
    """

    y, sr = librosa.load(audio_path, sr=sample_rate)

    if fmax is None:
        fmax = sr / 2

    requested_n_mfcc = n_mfcc + 1 if omit_zero_order else n_mfcc

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=requested_n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )

    if omit_zero_order:
        mfcc = mfcc[1:, :]

    # Shape: time_frames x n_mfcc
    return mfcc.T


# -----------------------------
# Save one feature file + label
# -----------------------------

def save_feature_with_label(
    audio_path: str | Path,
    output_dir: str | Path,
    n_mfcc: int,
) -> None:
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = audio_path.name.replace('&', '')  # Remove any leading/trailing special characters

    label_info = parse_audio_filename(audio_path.name)
    mfcc = extract_mfcc(audio_path, n_mfcc=n_mfcc)

    output_path = output_dir / f"{filename}.npz"

    # Convert dataclass label to plain JSON-compatible dict
    label_dict = asdict(label_info)

    np.savez_compressed(
        output_path,
        mfcc=mfcc,
        label_json=json.dumps(label_dict),
        class_label=label_info.label if label_info.label is not None else -1,
        recording_number=(
            label_info.recording_number
            if label_info.recording_number is not None
            else -1
        ),
    )

    print(f"Saved: {output_path}")
    print(f"MFCC shape: {mfcc.shape}")
    print(json.dumps(label_dict, indent=2))


# -----------------------------
# Process full dataset directory
# -----------------------------

def process_dataset(
    dataset_dir: str | Path,
    output_dir: str | Path,
    n_mfcc: int,
) -> None:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    audio_extensions = {".wav", ".flac", ".mp3", ".ogg"}

    for audio_path in dataset_dir.rglob("*"):
        if audio_path.suffix.lower() not in audio_extensions:
            continue

        try:
            save_feature_with_label(audio_path, output_dir, n_mfcc=n_mfcc)

        except Exception as error:
            print(f"Failed: {audio_path}")
            print(f"Reason: {error}")


# -----------------------------
# Example usage
# -----------------------------

if __name__ == "__main__":   
    process_dataset(
        dataset_dir="H:\Poberane\OneDrive_1_23.05.2026",
        output_dir="mfccs_10",
        n_mfcc=10
    )

    process_dataset(
        dataset_dir="H:\Poberane\OneDrive_2_23.05.2026",
        output_dir="mfccs_10",
        n_mfcc=10
    )

    process_dataset(
        dataset_dir="H:\Poberane\OneDrive_3_23.05.2026",
        output_dir="mfccs_10",
        n_mfcc=10
    )

    process_dataset(
        dataset_dir="H:\Poberane\OneDrive_1_23.05.2026",
        output_dir="mfccs_20",
        n_mfcc=20
    )

    process_dataset(
        dataset_dir="H:\Poberane\OneDrive_2_23.05.2026",
        output_dir="mfccs_20",
        n_mfcc=20
    )

    process_dataset(
        dataset_dir="H:\Poberane\OneDrive_3_23.05.2026",
        output_dir="mfccs_20",
        n_mfcc=20
    )

    process_dataset(
        dataset_dir="H:\Poberane\OneDrive_1_23.05.2026",
        output_dir="mfccs_40",
        n_mfcc=40
    )

    process_dataset(
        dataset_dir="H:\Poberane\OneDrive_2_23.05.2026",
        output_dir="mfccs_40",
        n_mfcc=40
    )

    process_dataset(
        dataset_dir="H:\Poberane\OneDrive_3_23.05.2026",
        output_dir="mfccs_40",
        n_mfcc=40
    )