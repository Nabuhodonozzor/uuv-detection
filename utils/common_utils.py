import json
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import keras
from keras import backend, callbacks, models
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split


CLASSES = [
    "ArtificialSignals", "BigPassengerShip", "Cargo", "FishBoat", "GreenCity",
    "KaiYan", "KaiYuan", "MotorBoat", "No7", "PoliceBoat", "QianDao",
    "SpeedBoat", "TheEarl", "TheKnight", "UUV", "Unknown", "WorkShip",
]


def configure_kaggle_access(secret_name: str = "COLAB") -> Path:
    """Use an existing Kaggle CLI credential or provision one from a Colab secret."""
    token_path = Path("/root/.kaggle/access_token")
    if token_path.exists():
        return token_path

    try:
        from google.colab import userdata
        from google.colab.userdata import TimeoutException
    except ImportError as exc:
        raise RuntimeError("Kaggle credential setup is only available in Google Colab.") from exc

    try:
        kaggle_token = userdata.get(secret_name)
    except TimeoutException as exc:
        raise RuntimeError(
            "Colab secrets are only available when the notebook runs from the Colab browser UI. "
            "Open the notebook in colab.research.google.com and run this cell there."
        ) from exc
    if not kaggle_token:
        raise RuntimeError(f"Set the {secret_name!r} Colab secret before downloading the dataset.")

    kaggle_dir = Path("/root/.kaggle")
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    token_path = kaggle_dir / "access_token"
    token_path.write_text(kaggle_token)
    token_path.chmod(0o600)
    return token_path


@dataclass
class DatasetSplits:
    train_data: np.ndarray
    train_labels_multi: np.ndarray
    train_labels_binary: np.ndarray
    val_data: np.ndarray
    val_labels_multi: np.ndarray
    val_labels_binary: np.ndarray
    test_data: np.ndarray
    test_labels_multi: np.ndarray
    test_labels_binary: np.ndarray
    classes: list[str]


@dataclass
class DatasetVariants:
    normal: DatasetSplits
    m: DatasetSplits
    w: DatasetSplits


@dataclass
class DatasetPool:
    cv_data: np.ndarray
    cv_labels_multi: np.ndarray
    cv_labels_binary: np.ndarray
    cv_fold_ids: np.ndarray
    test_data: np.ndarray
    test_labels_multi: np.ndarray
    test_labels_binary: np.ndarray
    classes: list[str]

    def fold(self, fold_index: int) -> DatasetSplits:
        if fold_index not in set(self.cv_fold_ids.tolist()):
            raise ValueError(f"Fold {fold_index} is not present in this dataset variant.")
        train_mask = self.cv_fold_ids != fold_index
        val_mask = self.cv_fold_ids == fold_index
        return DatasetSplits(
            train_data=self.cv_data[train_mask],
            train_labels_multi=self.cv_labels_multi[train_mask],
            train_labels_binary=self.cv_labels_binary[train_mask],
            val_data=self.cv_data[val_mask],
            val_labels_multi=self.cv_labels_multi[val_mask],
            val_labels_binary=self.cv_labels_binary[val_mask],
            test_data=self.test_data,
            test_labels_multi=self.test_labels_multi,
            test_labels_binary=self.test_labels_binary,
            classes=self.classes,
        )

    def final(self) -> DatasetSplits:
        empty_data = np.empty((0, *self.cv_data.shape[1:]), dtype=self.cv_data.dtype)
        empty_multi = np.empty((0, self.cv_labels_multi.shape[1]), dtype=np.float32)
        empty_binary = np.empty((0,), dtype=np.float32)
        return DatasetSplits(
            train_data=self.cv_data,
            train_labels_multi=self.cv_labels_multi,
            train_labels_binary=self.cv_labels_binary,
            val_data=empty_data,
            val_labels_multi=empty_multi,
            val_labels_binary=empty_binary,
            test_data=self.test_data,
            test_labels_multi=self.test_labels_multi,
            test_labels_binary=self.test_labels_binary,
            classes=self.classes,
        )


@dataclass
class CrossValidationDataset:
    normal: DatasetPool
    m: DatasetPool
    w: DatasetPool
    n_splits: int
    split_id: str
    n_mfcc: int

    def pools(self) -> dict[str, DatasetPool]:
        return {"normal": self.normal, "M": self.m, "W": self.w}

    def fold_variants(self, fold_index: int) -> DatasetVariants:
        if not 0 <= fold_index < self.n_splits:
            raise ValueError(f"fold_index must be in [0, {self.n_splits - 1}].")
        return DatasetVariants(
            normal=self.normal.fold(fold_index),
            m=self.m.fold(fold_index),
            w=self.w.fold(fold_index),
        )

    def final_variants(self) -> DatasetVariants:
        return DatasetVariants(
            normal=self.normal.final(),
            m=self.m.final(),
            w=self.w.final(),
        )


def prepare_spectrograms(spectrograms: np.ndarray) -> np.ndarray:
    spectrograms = np.asarray(spectrograms, dtype=np.float32)
    if spectrograms.ndim == 3:
        spectrograms = spectrograms[..., np.newaxis]
    if spectrograms.ndim != 4 or spectrograms.shape[-1] != 1:
        raise ValueError(f"Expected spectrograms with one channel, but received {spectrograms.shape}.")
    return spectrograms


def prepare_mfccs(mfccs: np.ndarray) -> np.ndarray:
    mfccs = np.asarray(mfccs, dtype=np.float32)
    if mfccs.ndim != 3:
        raise ValueError(f"Expected MFCC shape (samples, time_frames, coefficients), but received {mfccs.shape}.")
    return mfccs


def prepare_dataset(
    data_path: str | Path,
    test_size: float = 0.2,
    val_size: float = 0.2,
    uuv_filter: str | None = None,
    prepare_data: Callable[[np.ndarray], np.ndarray] = prepare_spectrograms,
) -> DatasetSplits:
    data_path = Path(data_path)
    x_values, y_values = [], []
    filter_info = f" (UUV filter: {uuv_filter})" if uuv_filter else ""
    print(f"Scanning feature data in: {data_path}{filter_info}")

    all_classes = sorted(CLASSES)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(all_classes)}
    uuv_idx = class_to_idx["UUV"]
    files_found = files_skipped = 0

    for root, _, files in os.walk(data_path):
        for filename in files:
            if not filename.endswith(".npz"):
                continue
            files_found += 1
            if uuv_filter and "UUV" in filename:
                match = re.search(r"UUV_[a-zA-Z]_([a-zA-Z])", filename)
                if match and match.group(1) != uuv_filter:
                    files_skipped += 1
                    continue
            try:
                data = np.load(Path(root) / filename, allow_pickle=True)
                labels = np.zeros(len(all_classes), dtype=np.int8)
                for target in json.loads(str(data["label_json"]))["targets"]:
                    if target["name"] in class_to_idx:
                        labels[class_to_idx[target["name"]]] = 1
                x_values.append(data["mfcc"])
                y_values.append(labels)
            except Exception as exc:
                print(f"Error loading {filename}: {exc}")

    print(f"Found .npz files: {files_found}, skipped: {files_skipped}")
    if not x_values:
        raise ValueError("No samples were loaded. Check data_path and filters.")

    x_values, y_values = np.asarray(x_values), np.asarray(y_values)
    print(f"Loaded samples: {len(x_values)}")
    print(f"X shape: {x_values.shape}")
    print(f"y shape: {y_values.shape}")
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        x_values, y_values, test_size=test_size, random_state=42
    )
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_val_data, train_val_labels, test_size=val_size, random_state=42
    )
    return DatasetSplits(
        train_data=prepare_data(train_data),
        train_labels_multi=np.asarray(train_labels, dtype=np.float32),
        train_labels_binary=np.asarray(train_labels[:, uuv_idx], dtype=np.float32),
        val_data=prepare_data(val_data),
        val_labels_multi=np.asarray(val_labels, dtype=np.float32),
        val_labels_binary=np.asarray(val_labels[:, uuv_idx], dtype=np.float32),
        test_data=prepare_data(test_data),
        test_labels_multi=np.asarray(test_labels, dtype=np.float32),
        test_labels_binary=np.asarray(test_labels[:, uuv_idx], dtype=np.float32),
        classes=all_classes,
    )


def prepare_dataset_variants(data_path: str | Path, test_size: float = 0.2, val_size: float = 0.2) -> DatasetVariants:
    return DatasetVariants(
        normal=prepare_dataset(data_path, test_size, val_size),
        m=prepare_dataset(data_path, test_size, val_size, uuv_filter="M"),
        w=prepare_dataset(data_path, test_size, val_size, uuv_filter="W"),
    )


def _filter_dataset_pool(
    pool: DatasetPool, cv_mask: np.ndarray, test_mask: np.ndarray
) -> DatasetPool:
    return DatasetPool(
        cv_data=pool.cv_data[cv_mask],
        cv_labels_multi=pool.cv_labels_multi[cv_mask],
        cv_labels_binary=pool.cv_labels_binary[cv_mask],
        cv_fold_ids=pool.cv_fold_ids[cv_mask],
        test_data=pool.test_data[test_mask],
        test_labels_multi=pool.test_labels_multi[test_mask],
        test_labels_binary=pool.test_labels_binary[test_mask],
        classes=pool.classes,
    )


def _validate_mfcc_archive(data: Any, data_path: Path) -> None:
    required_keys = {
        "cv_data", "cv_labels_multi", "cv_labels_binary", "cv_fold_ids",
        "cv_sample_ids", "cv_timestamps", "cv_uuv_middle", "cv_uuv_weak",
        "test_data", "test_labels_multi", "test_labels_binary", "test_sample_ids",
        "test_timestamps", "test_uuv_middle", "test_uuv_weak", "classes",
        "n_mfcc", "n_folds", "split_id", "split_ratios", "group_key",
        "format_version",
    }
    missing = sorted(required_keys - set(data.files))
    if missing:
        raise ValueError(f"MFCC archive {data_path} is missing keys: {missing}")
    if str(data["format_version"].item()) != "2":
        raise ValueError(
            f"Unsupported MFCC archive format {data['format_version'].item()!r}."
        )

    classes = data["classes"].astype(str).tolist()
    if classes != CLASSES:
        raise ValueError("The class order in the MFCC archive does not match CLASSES.")

    n_folds = int(data["n_folds"].item())
    if n_folds != 4:
        raise ValueError(f"Expected exactly 4 grouped CV folds, got {n_folds}.")
    if str(data["group_key"].item()) != "timestamp_raw":
        raise ValueError("MFCC folds must be grouped by timestamp_raw.")
    split_ratios = np.asarray(data["split_ratios"], dtype=np.float32)
    if split_ratios.shape != (n_folds, 3) or not np.allclose(
        np.sum(split_ratios, axis=1), 1.0
    ):
        raise ValueError(f"Invalid train/validation/test ratios: {split_ratios}")
    fold_ids = np.asarray(data["cv_fold_ids"], dtype=np.int8)
    if sorted(np.unique(fold_ids).tolist()) != list(range(n_folds)):
        raise ValueError(f"Expected fold IDs 0..{n_folds - 1}, got {np.unique(fold_ids)}.")

    cv_lengths = {
        len(data[key])
        for key in (
            "cv_data", "cv_labels_multi", "cv_labels_binary", "cv_fold_ids",
            "cv_sample_ids", "cv_timestamps", "cv_uuv_middle", "cv_uuv_weak",
        )
    }
    test_lengths = {
        len(data[key])
        for key in (
            "test_data", "test_labels_multi", "test_labels_binary", "test_sample_ids",
            "test_timestamps", "test_uuv_middle", "test_uuv_weak",
        )
    }
    if len(cv_lengths) != 1 or len(test_lengths) != 1:
        raise ValueError("MFCC archive arrays have inconsistent sample counts.")

    cv_ids = data["cv_sample_ids"].astype(str)
    test_ids = data["test_sample_ids"].astype(str)
    if len(np.unique(cv_ids)) != len(cv_ids) or len(np.unique(test_ids)) != len(test_ids):
        raise ValueError("MFCC archive contains duplicate sample IDs.")
    if set(cv_ids) & set(test_ids):
        raise ValueError("CV and test sample IDs overlap.")

    cv_timestamps = data["cv_timestamps"].astype(str)
    test_timestamps = data["test_timestamps"].astype(str)
    if set(cv_timestamps) & set(test_timestamps):
        raise ValueError("A timestamp group occurs in both CV and test data.")
    for timestamp in np.unique(cv_timestamps):
        timestamp_folds = np.unique(fold_ids[cv_timestamps == timestamp])
        if len(timestamp_folds) != 1:
            raise ValueError(f"Timestamp {timestamp} occurs in multiple CV folds.")

    n_mfcc = int(data["n_mfcc"].item())
    for key in ("cv_data", "test_data"):
        values = prepare_mfccs(data[key])
        if values.shape[-1] != n_mfcc:
            raise ValueError(
                f"{key} has {values.shape[-1]} coefficients, expected {n_mfcc}."
            )

    uuv_index = CLASSES.index("UUV")
    for prefix in ("cv", "test"):
        expected_binary = np.asarray(data[f"{prefix}_labels_multi"])[:, uuv_index]
        actual_binary = np.asarray(data[f"{prefix}_labels_binary"])
        if not np.array_equal(expected_binary, actual_binary):
            raise ValueError(f"{prefix} binary UUV labels do not match multilabel data.")


def prepare_mfcc_dataset_variants(data_path: str | Path) -> CrossValidationDataset:
    """Load persisted grouped folds; no random split is performed here."""
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"MFCC archive does not exist: {data_path}")

    with np.load(data_path, allow_pickle=False) as data:
        _validate_mfcc_archive(data, data_path)
        classes = data["classes"].astype(str).tolist()
        normal = DatasetPool(
            cv_data=prepare_mfccs(data["cv_data"]),
            cv_labels_multi=np.asarray(data["cv_labels_multi"], dtype=np.float32),
            cv_labels_binary=np.asarray(data["cv_labels_binary"], dtype=np.float32),
            cv_fold_ids=np.asarray(data["cv_fold_ids"], dtype=np.int8),
            test_data=prepare_mfccs(data["test_data"]),
            test_labels_multi=np.asarray(data["test_labels_multi"], dtype=np.float32),
            test_labels_binary=np.asarray(data["test_labels_binary"], dtype=np.float32),
            classes=classes,
        )
        cv_has_uuv = normal.cv_labels_binary.astype(bool)
        test_has_uuv = normal.test_labels_binary.astype(bool)
        middle = _filter_dataset_pool(
            normal,
            ~cv_has_uuv | np.asarray(data["cv_uuv_middle"], dtype=bool),
            ~test_has_uuv | np.asarray(data["test_uuv_middle"], dtype=bool),
        )
        weak = _filter_dataset_pool(
            normal,
            ~cv_has_uuv | np.asarray(data["cv_uuv_weak"], dtype=bool),
            ~test_has_uuv | np.asarray(data["test_uuv_weak"], dtype=bool),
        )
        n_splits = int(data["n_folds"].item())
        split_id = str(data["split_id"].item())
        n_mfcc = int(data["n_mfcc"].item())

    for variant_name, pool in {"normal": normal, "M": middle, "W": weak}.items():
        missing_folds = sorted(set(range(n_splits)) - set(pool.cv_fold_ids.tolist()))
        if missing_folds:
            raise ValueError(f"Variant {variant_name} is missing CV folds: {missing_folds}")
        if set(np.unique(pool.test_labels_binary).tolist()) != {0.0, 1.0}:
            raise ValueError(f"Variant {variant_name} test data must contain both binary classes.")
        for fold_index in range(n_splits):
            fold_data = pool.fold(fold_index)
            for split_name, labels in {
                "train": fold_data.train_labels_binary,
                "validation": fold_data.val_labels_binary,
            }.items():
                if set(np.unique(labels).tolist()) != {0.0, 1.0}:
                    raise ValueError(
                        f"Variant {variant_name} fold {fold_index} {split_name} "
                        "data must contain both binary classes."
                    )
        print(
            f"{variant_name}: CV={len(pool.cv_data)}, test={len(pool.test_data)}, "
            f"folds={np.bincount(pool.cv_fold_ids, minlength=n_splits).tolist()}"
        )

    print(f"Loaded MFCC-{n_mfcc} split {split_id} from {data_path}")
    return CrossValidationDataset(
        normal=normal,
        m=middle,
        w=weak,
        n_splits=n_splits,
        split_id=split_id,
        n_mfcc=n_mfcc,
    )


def train_keras_models_for_variants(
    models_by_variant: dict[str, models.Model], variants: DatasetVariants, model_type: str,
    epochs: int, batch_size: int, callback_factory: Callable[[], list[callbacks.Callback]],
) -> dict[str, callbacks.History]:
    histories = {}
    for variant_name, dataset in {"normal": variants.normal, "M": variants.m, "W": variants.w}.items():
        train_labels = dataset.train_labels_multi if model_type == "multilabel" else dataset.train_labels_binary
        val_labels = dataset.val_labels_multi if model_type == "multilabel" else dataset.val_labels_binary
        print(f"Training {model_type} model for variant: {variant_name}")
        histories[variant_name] = models_by_variant[variant_name].fit(
            dataset.train_data, train_labels, validation_data=(dataset.val_data, val_labels),
            epochs=epochs, batch_size=batch_size, callbacks=callback_factory(),
        )
    return histories


def predict_model_probabilities(model: Any, x_data: np.ndarray, binary: bool = False) -> np.ndarray:
    """Return probability-like scores from either a scikit-learn or Keras model."""
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(x_data))
    else:
        try:
            scores = np.asarray(model.predict(x_data, verbose=0))
        except TypeError:
            scores = np.asarray(model.predict(x_data))

    if binary and scores.ndim == 2 and scores.shape[1] == 2:
        return scores[:, 1]
    return scores


def _result_row(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_type: str,
    class_names: list[str],
) -> dict[str, float]:
    if model_type == "multilabel":
        y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        return {
            "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
            "uuv_precision": report["UUV"]["precision"],
            "uuv_recall": report["UUV"]["recall"],
            "uuv_f1": report["UUV"]["f1-score"],
            "uuv_support": report["UUV"]["support"],
        }

    y_true_binary = np.asarray(y_true).ravel().astype(int)
    y_pred_binary = (np.asarray(y_prob).ravel() >= 0.5).astype(int)
    report = classification_report(
        y_true_binary,
        y_pred_binary,
        labels=[0, 1],
        target_names=["No UUV", "UUV"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "uuv_precision": report["UUV"]["precision"],
        "uuv_recall": report["UUV"]["recall"],
        "uuv_f1": report["UUV"]["f1-score"],
        "uuv_support": report["UUV"]["support"],
    }


def _labels_for_model(dataset: DatasetSplits, split: str, model_type: str) -> np.ndarray:
    return getattr(dataset, f"{split}_labels_{'multi' if model_type == 'multilabel' else 'binary'}")


def cross_validate_keras_models_for_variants(
    model_builder: Callable[[tuple[int, ...], str], models.Model],
    dataset: CrossValidationDataset,
    model_type: str,
    epochs: int,
    batch_size: int,
    callback_factory: Callable[[], list[callbacks.Callback]],
    dataset_label: str,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, list[int]], dict[str, dict[str, list[float]]]]:
    """Run grouped CV without touching the persisted test set."""
    if model_type not in {"multilabel", "binary"}:
        raise ValueError("model_type must be either 'multilabel' or 'binary'")

    rows: list[dict[str, Any]] = []
    best_epochs = {variant_name: [] for variant_name in dataset.pools()}
    histories: dict[str, dict[str, list[float]]] = {}
    for fold_index in range(dataset.n_splits):
        for variant_offset, (variant_name, pool) in enumerate(dataset.pools().items()):
            fold_data = pool.fold(fold_index)
            backend.clear_session()
            keras.utils.set_random_seed(seed + 100 * fold_index + variant_offset)
            model = model_builder(fold_data.train_data.shape[1:], model_type)
            print(
                f"CV fold {fold_index + 1}/{dataset.n_splits}: "
                f"{model_type} {variant_name} {dataset_label}"
            )
            history = model.fit(
                fold_data.train_data,
                _labels_for_model(fold_data, "train", model_type),
                validation_data=(
                    fold_data.val_data,
                    _labels_for_model(fold_data, "val", model_type),
                ),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callback_factory(),
                verbose=2,
            )
            best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
            best_epochs[variant_name].append(best_epoch)
            history_key = f"{model_type}_fold_{fold_index}_{variant_name}"
            histories[history_key] = history.history
            y_val = _labels_for_model(fold_data, "val", model_type)
            y_prob = predict_model_probabilities(
                model, fold_data.val_data, binary=model_type == "binary"
            )
            rows.append(
                {
                    "Dataset": dataset_label,
                    "ModelType": model_type,
                    "Variant": variant_name,
                    "Fold": fold_index,
                    "BestEpoch": best_epoch,
                    **_result_row(y_val, y_prob, model_type, fold_data.classes),
                }
            )
            del model
            backend.clear_session()
    return pd.DataFrame(rows), best_epochs, histories


def train_final_keras_models_for_variants(
    model_builder: Callable[[tuple[int, ...], str], models.Model],
    dataset: CrossValidationDataset,
    model_type: str,
    best_epochs: dict[str, list[int]],
    batch_size: int,
    seed: int = 42,
) -> tuple[dict[str, models.Model], dict[str, callbacks.History]]:
    """Train fresh final models on all non-test data using median CV epochs."""
    final_models: dict[str, models.Model] = {}
    histories: dict[str, callbacks.History] = {}
    for variant_offset, (variant_name, pool) in enumerate(dataset.pools().items()):
        final_data = pool.final()
        final_epochs = max(1, int(round(float(np.median(best_epochs[variant_name])))))
        backend.clear_session()
        keras.utils.set_random_seed(seed + 10_000 + variant_offset)
        model = model_builder(final_data.train_data.shape[1:], model_type)
        print(
            f"Final training: {model_type} {variant_name}, "
            f"samples={len(final_data.train_data)}, epochs={final_epochs}"
        )
        histories[variant_name] = model.fit(
            final_data.train_data,
            _labels_for_model(final_data, "train", model_type),
            epochs=final_epochs,
            batch_size=batch_size,
            verbose=2,
        )
        final_models[variant_name] = model
    return final_models, histories


def cross_validate_sklearn_models_for_variants(
    model_builder: Callable[[str], Any],
    dataset: CrossValidationDataset,
    model_type: str,
    dataset_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold_index in range(dataset.n_splits):
        for variant_name, pool in dataset.pools().items():
            fold_data = pool.fold(fold_index)
            model = model_builder(model_type)
            print(
                f"CV fold {fold_index + 1}/{dataset.n_splits}: "
                f"{model_type} {variant_name} {dataset_label}"
            )
            model.fit(
                fold_data.train_data,
                _labels_for_model(fold_data, "train", model_type),
            )
            y_val = _labels_for_model(fold_data, "val", model_type)
            y_prob = predict_model_probabilities(
                model, fold_data.val_data, binary=model_type == "binary"
            )
            rows.append(
                {
                    "Dataset": dataset_label,
                    "ModelType": model_type,
                    "Variant": variant_name,
                    "Fold": fold_index,
                    **_result_row(y_val, y_prob, model_type, fold_data.classes),
                }
            )
    return pd.DataFrame(rows)


def train_final_sklearn_models_for_variants(
    model_builder: Callable[[str], Any],
    dataset: CrossValidationDataset,
    model_type: str,
) -> dict[str, Any]:
    final_models = {}
    for variant_name, pool in dataset.pools().items():
        final_data = pool.final()
        model = model_builder(model_type)
        print(
            f"Final training: {model_type} {variant_name}, "
            f"samples={len(final_data.train_data)}"
        )
        model.fit(
            final_data.train_data,
            _labels_for_model(final_data, "train", model_type),
        )
        final_models[variant_name] = model
    return final_models


def summarize_cross_validation(results: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["Dataset", "ModelType", "Variant"]
    metric_columns = [
        column
        for column in results.select_dtypes(include=[np.number]).columns
        if column not in {"Fold", "BestEpoch"}
    ]
    summary = results.groupby(group_columns)[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def evaluate_multilabel_model(model: Any, x_test: np.ndarray, y_test: np.ndarray, class_names: list[str], title: str) -> dict:
    print(f"\n{'=' * 50}\n--- Evaluation for: {title} ---\n{'=' * 50}\n")
    y_prob = predict_model_probabilities(model, x_test)
    y_pred = (y_prob >= 0.5).astype(int)
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    print("Micro F1:", f1_score(y_test, y_pred, average="micro", zero_division=0))
    print("Macro F1:", f1_score(y_test, y_pred, average="macro", zero_division=0))
    print("Samples F1:", f1_score(y_test, y_pred, average="samples", zero_division=0))
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    return {
        "report": report,
        "micro_f1": f1_score(y_test, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "samples_f1": f1_score(y_test, y_pred, average="samples", zero_division=0),
    }


def evaluate_binary_model(model: Any, x_test: np.ndarray, y_test: np.ndarray, title: str) -> dict:
    print(f"\n{'=' * 50}\n--- Evaluation for: {title} ---\n{'=' * 50}\n")
    y_prob = predict_model_probabilities(model, x_test, binary=True).ravel()
    y_true, y_pred = np.asarray(y_test).ravel().astype(int), (y_prob >= 0.5).astype(int)
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=["No UUV", "UUV"], zero_division=0))
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=["No UUV", "UUV"], output_dict=True, zero_division=0)
    return {"Model": title, **{key: report["UUV"][key] for key in ("precision", "recall", "f1-score", "support")}}


def evaluate_models_for_variants(models_by_variant: dict[str, Any], variants: DatasetVariants, model_type: str, dataset_label: str) -> pd.DataFrame:
    rows = []
    for variant_name, dataset in {"normal": variants.normal, "M": variants.m, "W": variants.w}.items():
        title = f"{model_type.title()} {variant_name} {dataset_label}"
        if model_type == "multilabel":
            result = evaluate_multilabel_model(models_by_variant[variant_name], dataset.test_data, dataset.test_labels_multi, dataset.classes, title)
            rows.append({
                "Model": title,
                **{key: result["report"]["UUV"][key] for key in ("precision", "recall", "f1-score", "support")},
                "micro_f1": result["micro_f1"],
                "macro_f1": result["macro_f1"],
                "samples_f1": result["samples_f1"],
            })
        else:
            rows.append(evaluate_binary_model(models_by_variant[variant_name], dataset.test_data, dataset.test_labels_binary, title))
    return pd.DataFrame(rows)


def plot_training_histories(histories: dict[str, callbacks.History], title: str) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(histories), figsize=(6 * len(histories), 8))
    if len(histories) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    fig.suptitle(title, fontsize=16)
    for idx, (variant_name, history) in enumerate(histories.items()):
        axes[0, idx].plot(history.history["loss"], label="Train Loss")
        if "val_loss" in history.history:
            axes[0, idx].plot(history.history["val_loss"], label="Val Loss")
        axes[0, idx].set_title(f"{variant_name} - Loss")
        axes[0, idx].legend()
        axes[1, idx].plot(history.history["roc_auc"], label="Train ROC AUC")
        if "val_roc_auc" in history.history:
            axes[1, idx].plot(history.history["val_roc_auc"], label="Val ROC AUC")
        axes[1, idx].set_title(f"{variant_name} - ROC AUC")
        axes[1, idx].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def save_keras_artifacts(
    save_dir: str | Path, dataset_key: str, multilabel_models: dict[str, models.Model], binary_models: dict[str, models.Model],
    multilabel_histories: dict[str, callbacks.History], binary_histories: dict[str, callbacks.History],
    multilabel_results: pd.DataFrame, binary_results: pd.DataFrame,
    cv_multilabel_results: pd.DataFrame | None = None,
    cv_binary_results: pd.DataFrame | None = None,
    cv_histories: dict[str, dict[str, list[float]]] | None = None,
    split_metadata: dict[str, Any] | None = None,
) -> Path:
    import pickle
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for variant_name, model in multilabel_models.items():
        model.save(save_dir / f"model_{dataset_key}_{variant_name}.keras")
    for variant_name, model in binary_models.items():
        model.save(save_dir / f"bin_model_{dataset_key}_{variant_name}.keras")
    histories = {**{f"history_{dataset_key}_{key}": value.history for key, value in multilabel_histories.items()}, **{f"bin_history_{dataset_key}_{key}": value.history for key, value in binary_histories.items()}}
    with open(save_dir / f"training_histories_{dataset_key}.pkl", "wb") as file_obj:
        pickle.dump(histories, file_obj)
    multilabel_results.to_csv(save_dir / f"uuv_evaluation_results_{dataset_key}.csv", index=False)
    binary_results.to_csv(save_dir / f"binary_uuv_evaluation_results_{dataset_key}.csv", index=False)
    if cv_multilabel_results is not None and cv_binary_results is not None:
        cv_results = pd.concat([cv_multilabel_results, cv_binary_results], ignore_index=True)
        cv_results.to_csv(save_dir / f"cross_validation_results_{dataset_key}.csv", index=False)
        summarize_cross_validation(cv_results).to_csv(
            save_dir / f"cross_validation_summary_{dataset_key}.csv", index=False
        )
    if cv_histories is not None:
        with open(save_dir / f"cross_validation_histories_{dataset_key}.pkl", "wb") as file_obj:
            pickle.dump(cv_histories, file_obj)
    if split_metadata is not None:
        with open(save_dir / f"dataset_split_{dataset_key}.json", "w", encoding="utf-8") as file_obj:
            json.dump(split_metadata, file_obj, indent=2, sort_keys=True)
    return save_dir


def zip_artifacts(save_dir: str | Path, archive_name: str | Path) -> Path:
    return Path(shutil.make_archive(str(Path(archive_name).with_suffix("")), "zip", save_dir))


def extract_zip(zip_path: str | Path, extract_to: str | Path) -> Path:
    zip_path, extract_to = Path(zip_path), Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        top_level_dirs = {Path(zip_info.filename).parts[0] for zip_info in zip_ref.infolist() if Path(zip_info.filename).parts and (zip_info.is_dir() or len(Path(zip_info.filename).parts) > 1)}
        zip_ref.extractall(extract_to)
    return extract_to / next(iter(top_level_dirs)) if len(top_level_dirs) == 1 else extract_to
