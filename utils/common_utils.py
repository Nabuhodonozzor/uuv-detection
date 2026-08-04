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
from keras import callbacks, models
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


def prepare_mfcc_dataset_variants(data_path: str | Path, test_size: float = 0.2, val_size: float = 0.2) -> DatasetVariants:
    return DatasetVariants(
        normal=prepare_dataset(data_path, test_size, val_size, prepare_data=prepare_mfccs),
        m=prepare_dataset(data_path, test_size, val_size, uuv_filter="M", prepare_data=prepare_mfccs),
        w=prepare_dataset(data_path, test_size, val_size, uuv_filter="W", prepare_data=prepare_mfccs),
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


def evaluate_multilabel_model(model: Any, x_test: np.ndarray, y_test: np.ndarray, class_names: list[str], title: str) -> dict:
    print(f"\n{'=' * 50}\n--- Evaluation for: {title} ---\n{'=' * 50}\n")
    y_prob = predict_model_probabilities(model, x_test)
    y_pred = (y_prob >= 0.5).astype(int)
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    print("Micro F1:", f1_score(y_test, y_pred, average="micro", zero_division=0))
    print("Macro F1:", f1_score(y_test, y_pred, average="macro", zero_division=0))
    print("Samples F1:", f1_score(y_test, y_pred, average="samples", zero_division=0))
    return classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)


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
            report = evaluate_multilabel_model(models_by_variant[variant_name], dataset.test_data, dataset.test_labels_multi, dataset.classes, title)
            rows.append({"Model": title, **{key: report["UUV"][key] for key in ("precision", "recall", "f1-score", "support")}})
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
        axes[0, idx].plot(history.history["val_loss"], label="Val Loss")
        axes[0, idx].set_title(f"{variant_name} - Loss")
        axes[0, idx].legend()
        axes[1, idx].plot(history.history["roc_auc"], label="Train ROC AUC")
        axes[1, idx].plot(history.history["val_roc_auc"], label="Val ROC AUC")
        axes[1, idx].set_title(f"{variant_name} - ROC AUC")
        axes[1, idx].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def save_keras_artifacts(
    save_dir: str | Path, dataset_key: str, multilabel_models: dict[str, models.Model], binary_models: dict[str, models.Model],
    multilabel_histories: dict[str, callbacks.History], binary_histories: dict[str, callbacks.History],
    multilabel_results: pd.DataFrame, binary_results: pd.DataFrame,
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
