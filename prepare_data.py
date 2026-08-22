"""Single linear data-preparation pipeline for binary UUV detection.

Replaces the MFCC-10/20/40 x normal/M/W matrix in ``load_data.py`` with one
path: manifest -> recording-level split -> log-mel features -> train-fitted
normalisation -> cached ``.npz`` per split.

Run it once; training notebooks then just load the arrays.

    python prepare_data.py --dataset-dir /path/to/clips --output-dir prepared

Design notes (each of these fixes something diagnosed in REPORT.md):

* **Split before anything else, at the recording level.** The clips are already
  segmented upstream, so a clip's source recording is only recoverable from the
  timestamp prefix in its filename. Every clip sharing a timestamp goes to the
  same split, whole. Splitting a flat clip list instead - what
  ``common_utils.prepare_dataset`` still does - puts adjacent seconds of one
  recording in both train and test and inflates accuracy to ~99%.

* **A sample rate that matches the signal.** The old MFCC settings
  (native 52734 Hz, ``n_fft=2048``, ``n_mels=256``) gave 25.75 Hz FFT bins while
  the lowest 66 mel filters are narrower than that - so the whole 0-1090 Hz
  band, where UUV and ship signatures live, was resolved by degenerate filters.
  Resampling to 16 kHz with 64 mel bands puts every filter above one bin wide
  and shrinks the input from 618x40 to 187x64.

* **One feature type.** Log-mel spectrogram only. MFCC is a lossy rotation of
  the same thing; keeping three MFCC resolutions multiplied the experiment
  matrix without ever isolating a cause.

* **Normalisation fitted on train, reused verbatim.** Per-mel-band mean/std from
  the training split only, saved to disk and applied unchanged to val and test.

* **Binary task only.** 7 of the 21 multilabel classes never occur in the data,
  so macro-F1 over them is meaningless. If you want the multilabel task back,
  add the label matrix here - do not fork the pipeline.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from load_data import AUDIO_EXTENSIONS, parse_audio_filename

# --- Feature configuration ---------------------------------------------------
# Changing any of these invalidates cached features; the values are written to
# config.json next to the arrays so a stale cache is detectable.
SAMPLE_RATE = 16_000        # resample target; source is 52734 Hz
N_FFT = 2048                # 128 ms window, 7.81 Hz bins
HOP_LENGTH = 256            # 16 ms hop -> 187 frames per 3 s clip
N_MELS = 64
FMIN = 10.0                 # below this is hydrophone/platform noise
FMAX = 8000.0               # Nyquist at SAMPLE_RATE
CLIP_SECONDS = 3.0

SPLIT_FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}


@dataclass
class Clip:
    """One 3-second clip plus the metadata needed to split and label it."""

    path: Path
    recording_id: str   # timestamp prefix - the grouping key
    is_uuv: bool
    audibility: str     # "strong" / "middle" / "weak" / "unknown", for slicing results


# --- Step 1: manifest --------------------------------------------------------


def build_manifest(dataset_dirs: Iterable[str | Path]) -> list[Clip]:
    """List every clip with its source-recording id and binary UUV label."""
    clips: list[Clip] = []
    for dataset_dir in dataset_dirs:
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
        for path in sorted(dataset_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            label = parse_audio_filename(path.name)
            uuv_targets = [t for t in label.targets if t.name == "UUV"]
            clips.append(
                Clip(
                    path=path,
                    recording_id=label.timestamp_raw,
                    is_uuv=bool(uuv_targets),
                    audibility=uuv_targets[0].audibility if uuv_targets else "none",
                )
            )
    if not clips:
        raise ValueError("No audio files found.")
    return clips


# --- Step 2: recording-level split -------------------------------------------


def split_recordings(clips: list[Clip], seed: int = 42) -> dict[str, str]:
    """Assign whole recordings to train/val/test. Returns recording_id -> split.

    Greedy and deterministic: recordings are shuffled once, then each is placed
    in whichever split is furthest below its target share of *UUV-positive*
    clips (falling back to total clips for UUV-free recordings). Positives are
    the scarce resource, so they drive the balance.

    This is deliberately simpler than ``load_data.assign_test_and_folds``, which
    ran 2048 randomised restarts optimising a 23-column balance objective. That
    search made the test set a *chosen*, not a random, sample of recordings -
    which is why its scores sit well above the cross-validated ones.
    """
    by_recording: dict[str, list[Clip]] = defaultdict(list)
    for clip in clips:
        by_recording[clip.recording_id].append(clip)

    positives = {r: sum(c.is_uuv for c in g) for r, g in by_recording.items()}
    totals = {r: len(g) for r, g in by_recording.items()}

    uuv_recordings = [r for r, n in positives.items() if n > 0]
    if len(uuv_recordings) < 3:
        raise ValueError(
            f"Only {len(uuv_recordings)} UUV-positive recordings exist; a "
            "train/val/test split cannot give each split a positive example."
        )
    if len(uuv_recordings) < 15:
        print(
            f"WARNING: only {len(uuv_recordings)} UUV-positive source recordings.\n"
            "  Held-out scores will swing by tens of F1 points depending on which\n"
            "  recordings land in test. Treat any single number as provisional and\n"
            "  report grouped cross-validation with a spread instead."
        )

    rng = np.random.default_rng(seed)
    # Place UUV-positive recordings first (scarce), then the rest.
    order = [
        sorted(uuv_recordings)[i] for i in rng.permutation(len(uuv_recordings))
    ]
    others = sorted(set(by_recording) - set(uuv_recordings))
    order += [others[i] for i in rng.permutation(len(others))]

    total_positive = sum(positives.values())
    total_clips = sum(totals.values())
    assignment: dict[str, str] = {}
    got_positive = {name: 0 for name in SPLIT_FRACTIONS}
    got_clips = {name: 0 for name in SPLIT_FRACTIONS}

    for recording_id in order:
        # Deficit = how far this split is below its target, as a fraction.
        if positives[recording_id] > 0:
            deficit = {
                name: fraction * total_positive - got_positive[name]
                for name, fraction in SPLIT_FRACTIONS.items()
            }
        else:
            deficit = {
                name: fraction * total_clips - got_clips[name]
                for name, fraction in SPLIT_FRACTIONS.items()
            }
        chosen = max(deficit, key=lambda name: deficit[name])
        assignment[recording_id] = chosen
        got_positive[chosen] += positives[recording_id]
        got_clips[chosen] += totals[recording_id]

    for name in SPLIT_FRACTIONS:
        if got_positive[name] == 0:
            raise ValueError(f"Split {name!r} received no UUV-positive clips.")
    return assignment


# --- Step 3: features --------------------------------------------------------


def _check_mel_resolution() -> None:
    """Fail loudly if the mel filterbank is finer than the FFT can resolve.

    This is the exact defect the old MFCC settings had: mel filters narrower
    than one FFT bin capture zero or one bin each, so the low-frequency band is
    not actually resolved no matter how many filters you ask for.
    """
    def hz_to_mel(f: float) -> float:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    bin_hz = SAMPLE_RATE / N_FFT
    low, high = hz_to_mel(FMIN), hz_to_mel(FMAX)
    step = (high - low) / (N_MELS + 1)
    degenerate = sum(
        1
        for i in range(1, N_MELS + 1)
        if (mel_to_hz(low + step * (i + 1)) - mel_to_hz(low + step * (i - 1))) / 2 < bin_hz
    )
    if degenerate:
        raise ValueError(
            f"{degenerate}/{N_MELS} mel filters are narrower than the {bin_hz:.2f} Hz "
            f"FFT bin spacing. Raise N_FFT, lower N_MELS, or lower SAMPLE_RATE."
        )


def extract_logmel(path: Path) -> np.ndarray:
    """Log-mel spectrogram, shape (frames, N_MELS), dB scale.

    ``librosa.load(sr=SAMPLE_RATE)`` resamples with an anti-aliasing filter, so
    the 52734 Hz source is low-passed before decimation - no aliasing of
    high-frequency noise into the band we care about.
    """
    import librosa  # imported here so the manifest/split steps need no audio deps

    y, sr = librosa.load(path, sr=SAMPLE_RATE)

    # Pad or trim to an exact clip length so every feature array has one shape.
    # The old pipeline relied on all clips happening to be identical and only
    # discovered mismatches at np.stack time, with an unhelpful error.
    target = int(round(CLIP_SECONDS * sr))
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0,
    )
    # Fixed dB reference (not per-clip max): a per-clip reference would erase
    # absolute level, which is a real cue for "is a source present".
    log_mel = librosa.power_to_db(mel, ref=1.0, top_db=None)
    return np.asarray(log_mel.T, dtype=np.float32)


# --- Steps 4-6: normalise, cache ---------------------------------------------


def main(
    dataset_dirs: Iterable[str | Path],
    output_dir: str | Path,
    seed: int = 42,
) -> None:
    _check_mel_resolution()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clips = build_manifest(dataset_dirs)
    assignment = split_recordings(clips, seed=seed)

    # --- manifest CSV: readable, greppable, and enough to rebuild any split ---
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("split,recording_id,is_uuv,audibility,path\n")
        for clip in clips:
            handle.write(
                f"{assignment[clip.recording_id]},{clip.recording_id},"
                f"{int(clip.is_uuv)},{clip.audibility},{clip.path.as_posix()}\n"
            )

    print(f"{len(clips)} clips from {len(assignment)} recordings")
    for name in SPLIT_FRACTIONS:
        split_clips = [c for c in clips if assignment[c.recording_id] == name]
        recordings = {c.recording_id for c in split_clips}
        uuv_recordings = {c.recording_id for c in split_clips if c.is_uuv}
        positives = sum(c.is_uuv for c in split_clips)
        print(
            f"  {name:<5} clips={len(split_clips):>6} "
            f"(UUV {positives}, {positives / max(len(split_clips), 1):.1%})  "
            f"recordings={len(recordings)} (UUV-positive {len(uuv_recordings)})"
        )

    # --- features, one split at a time so peak memory stays bounded ---
    features: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for name in SPLIT_FRACTIONS:
        split_clips = [c for c in clips if assignment[c.recording_id] == name]
        stack = np.empty(
            (len(split_clips), 1 + int(CLIP_SECONDS * SAMPLE_RATE) // HOP_LENGTH, N_MELS),
            dtype=np.float32,
        )
        for index, clip in enumerate(split_clips):
            if index % 250 == 0:
                print(f"  {name}: {index}/{len(split_clips)}")
            feature = extract_logmel(clip.path)
            if feature.shape != stack.shape[1:]:
                raise ValueError(
                    f"{clip.path.name} produced {feature.shape}, expected "
                    f"{stack.shape[1:]}. Clip lengths are inconsistent."
                )
            stack[index] = feature
        features[name] = stack
        labels[name] = np.asarray([c.is_uuv for c in split_clips], dtype=np.float32)

    # --- normalisation fitted on TRAIN ONLY, then applied unchanged ---
    # Per-mel-band statistics over (clips, frames): each frequency band gets its
    # own mean/std, preserving the relative shape of the spectrum across bands.
    # (The models' current first layer, LayerNormalization(axis=-1), instead
    # normalises within each frame across bands, which flattens exactly the
    # spectral shape that distinguishes a UUV from a boat.)
    mean = features["train"].mean(axis=(0, 1), keepdims=True)
    std = features["train"].std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-6)
    for name in features:
        features[name] = (features[name] - mean) / std

    config = {
        "sample_rate": SAMPLE_RATE, "n_fft": N_FFT, "hop_length": HOP_LENGTH,
        "n_mels": N_MELS, "fmin": FMIN, "fmax": FMAX,
        "clip_seconds": CLIP_SECONDS, "seed": seed,
        "feature": "log_mel_db", "group_key": "recording_timestamp",
        "input_shape": list(features["train"].shape[1:]),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "norm_stats.npz", mean=mean.squeeze(), std=std.squeeze()
    )
    for name in features:
        split_clips = [c for c in clips if assignment[c.recording_id] == name]
        np.savez_compressed(
            output_dir / f"{name}.npz",
            x=features[name],
            y=labels[name],
            recording_id=np.asarray([c.recording_id for c in split_clips]),
            audibility=np.asarray([c.audibility for c in split_clips]),
        )
        print(f"Saved {output_dir / f'{name}.npz'}  {features[name].shape}")
    print(f"Saved {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", action="append", required=True)
    parser.add_argument("--output-dir", default="prepared")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.dataset_dir, args.output_dir, seed=args.seed)
