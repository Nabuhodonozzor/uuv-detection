import json
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from keras import callbacks, layers, metrics, models, optimizers
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split


CLASSES = [
    "ArtificialSignals",
    "BigPassengerShip",
    "Cargo",
    "FishBoat",
    "GreenCity",
    "KaiYan",
    "KaiYuan",
    "MotorBoat",
    "No7",
    "PoliceBoat",
    "QianDao",
    "SpeedBoat",
    "TheEarl",
    "TheKnight",
    "UUV",
    "Unknown",
    "WorkShip",
]


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
        raise ValueError(
            "Expected shape (samples, time_frames, n_mels) or "
            f"(samples, time_frames, n_mels, 1), but received {spectrograms.shape}."
        )

    return spectrograms


def prepare_dataset(
    data_path: str | Path,
    test_size: float = 0.2,
    val_size: float = 0.2,
    uuv_filter: str | None = None,
) -> DatasetSplits:
    data_path = Path(data_path)
    x_values = []
    y_values = []

    filter_info = f" (UUV filter: {uuv_filter})" if uuv_filter else ""
    print(f"Scanning spectrogram data in: {data_path}{filter_info}")

    all_classes = sorted(CLASSES)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(all_classes)}
    num_classes = len(all_classes)
    uuv_idx = class_to_idx["UUV"]

    files_found = 0
    files_skipped = 0

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

            filepath = Path(root) / filename

            try:
                data = np.load(filepath, allow_pickle=True)
                spectrogram = data["mfcc"]
                label_info = json.loads(str(data["label_json"]))

                labels = np.zeros(num_classes, dtype=np.int8)
                for target in label_info["targets"]:
                    target_name = target["name"]
                    if target_name in class_to_idx:
                        labels[class_to_idx[target_name]] = 1

                x_values.append(spectrogram)
                y_values.append(labels)
            except Exception as exc:
                print(f"Error loading {filename}: {exc}")

    print(f"Found .npz files: {files_found}, skipped: {files_skipped}")

    if not x_values:
        raise ValueError("No samples were loaded. Check data_path and filters.")

    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)

    print(f"Loaded samples: {len(x_values)}")
    print(f"X shape: {x_values.shape}")
    print(f"y shape: {y_values.shape}")

    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        x_values,
        y_values,
        test_size=test_size,
        random_state=42,
    )

    train_data, val_data, train_labels, val_labels = train_test_split(
        train_val_data,
        train_val_labels,
        test_size=val_size,
        random_state=42,
    )

    return DatasetSplits(
        train_data=prepare_spectrograms(train_data),
        train_labels_multi=np.asarray(train_labels, dtype=np.float32),
        train_labels_binary=np.asarray(train_labels[:, uuv_idx], dtype=np.float32),
        val_data=prepare_spectrograms(val_data),
        val_labels_multi=np.asarray(val_labels, dtype=np.float32),
        val_labels_binary=np.asarray(val_labels[:, uuv_idx], dtype=np.float32),
        test_data=prepare_spectrograms(test_data),
        test_labels_multi=np.asarray(test_labels, dtype=np.float32),
        test_labels_binary=np.asarray(test_labels[:, uuv_idx], dtype=np.float32),
        classes=all_classes,
    )


def prepare_dataset_variants(
    data_path: str | Path,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> DatasetVariants:
    return DatasetVariants(
        normal=prepare_dataset(data_path, test_size=test_size, val_size=val_size),
        m=prepare_dataset(data_path, test_size=test_size, val_size=val_size, uuv_filter="M"),
        w=prepare_dataset(data_path, test_size=test_size, val_size=val_size, uuv_filter="W"),
    )


def build_spectrogram_cnn(
    input_shape: tuple[int, int, int],
    model_type: str = "multilabel",
    num_classes: int = len(CLASSES),
) -> models.Model:
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Rescaling(scale=1.0 / 80.0, offset=1.0),
            layers.Conv2D(filters=32, kernel_size=(5, 5), padding="same", use_bias=False, name="conv_1"),
            layers.BatchNormalization(name="batch_norm_1"),
            layers.Activation("relu", name="relu_1"),
            layers.MaxPooling2D(pool_size=(2, 2), name="pool_1"),
            layers.SpatialDropout2D(0.15, name="spatial_dropout_1"),
            layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", use_bias=False, name="conv_2"),
            layers.BatchNormalization(name="batch_norm_2"),
            layers.Activation("relu", name="relu_2"),
            layers.MaxPooling2D(pool_size=(2, 2), name="pool_2"),
            layers.SpatialDropout2D(0.20, name="spatial_dropout_2"),
            layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", use_bias=False, name="conv_3"),
            layers.BatchNormalization(name="batch_norm_3"),
            layers.Activation("relu", name="relu_3"),
            layers.MaxPooling2D(pool_size=(2, 2), name="pool_3"),
            layers.SpatialDropout2D(0.30, name="spatial_dropout_3"),
            layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", use_bias=False, name="conv_4"),
            layers.BatchNormalization(name="batch_norm_4"),
            layers.Activation("relu", name="relu_4"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.40),
        ]
    )

    if model_type == "multilabel":
        model.add(layers.Dense(num_classes, activation="sigmoid"))
    elif model_type == "binary":
        model.add(layers.Dense(1, activation="sigmoid"))
    else:
        raise ValueError("model_type must be either 'multilabel' or 'binary'")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5),
            metrics.AUC(name="roc_auc", curve="ROC"),
            metrics.AUC(name="pr_auc", curve="PR"),
            metrics.Precision(name="precision", thresholds=0.5),
            metrics.Recall(name="recall", thresholds=0.5),
        ],
    )

    return model


def get_cnn_callbacks() -> list[callbacks.Callback]:
    return [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]


def build_models_for_variants(variants: DatasetVariants, model_type: str) -> dict[str, models.Model]:
    return {
        "normal": build_spectrogram_cnn(variants.normal.train_data.shape[1:], model_type=model_type),
        "M": build_spectrogram_cnn(variants.m.train_data.shape[1:], model_type=model_type),
        "W": build_spectrogram_cnn(variants.w.train_data.shape[1:], model_type=model_type),
    }


def train_models_for_variants(
    models_by_variant: dict[str, models.Model],
    variants: DatasetVariants,
    model_type: str,
    epochs: int = 50,
    batch_size: int = 32,
) -> dict[str, callbacks.History]:
    histories = {}
    variant_splits = {"normal": variants.normal, "M": variants.m, "W": variants.w}

    for variant_name, dataset in variant_splits.items():
        train_labels = dataset.train_labels_multi if model_type == "multilabel" else dataset.train_labels_binary
        val_labels = dataset.val_labels_multi if model_type == "multilabel" else dataset.val_labels_binary

        print(f"Training {model_type} model for variant: {variant_name}")
        histories[variant_name] = models_by_variant[variant_name].fit(
            dataset.train_data,
            train_labels,
            validation_data=(dataset.val_data, val_labels),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=get_cnn_callbacks(),
        )

    return histories


def evaluate_multilabel_model(
    model: models.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    title: str,
) -> dict:
    print(f"\n{'=' * 50}")
    print(f"--- Evaluation for: {title} ---")
    print(f"{'=' * 50}\n")

    y_prob = model.predict(x_test, verbose=0)
    y_pred = (y_prob >= 0.5).astype(int)

    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    print("Micro F1:", f1_score(y_test, y_pred, average="micro", zero_division=0))
    print("Macro F1:", f1_score(y_test, y_pred, average="macro", zero_division=0))
    print("Samples F1:", f1_score(y_test, y_pred, average="samples", zero_division=0))

    return classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


def evaluate_binary_model(
    model: models.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    title: str,
) -> dict:
    print(f"\n{'=' * 50}")
    print(f"--- Evaluation for: {title} ---")
    print(f"{'=' * 50}\n")

    y_prob = model.predict(x_test, verbose=0).ravel()
    y_true = np.asarray(y_test).ravel().astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=["No UUV", "UUV"], zero_division=0))

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["No UUV", "UUV"],
        output_dict=True,
        zero_division=0,
    )
    uuv_metrics = report["UUV"]

    return {
        "Model": title,
        "precision": uuv_metrics["precision"],
        "recall": uuv_metrics["recall"],
        "f1-score": uuv_metrics["f1-score"],
        "support": uuv_metrics["support"],
    }


def evaluate_models_for_variants(
    models_by_variant: dict[str, models.Model],
    variants: DatasetVariants,
    model_type: str,
    dataset_label: str,
) -> pd.DataFrame:
    rows = []
    variant_splits = {"normal": variants.normal, "M": variants.m, "W": variants.w}

    for variant_name, dataset in variant_splits.items():
        title = f"{model_type.title()} {variant_name} {dataset_label}"
        if model_type == "multilabel":
            report = evaluate_multilabel_model(
                models_by_variant[variant_name],
                dataset.test_data,
                dataset.test_labels_multi,
                dataset.classes,
                title,
            )
            uuv_metrics = report["UUV"]
            rows.append(
                {
                    "Model": title,
                    "precision": uuv_metrics["precision"],
                    "recall": uuv_metrics["recall"],
                    "f1-score": uuv_metrics["f1-score"],
                    "support": uuv_metrics["support"],
                }
            )
        else:
            rows.append(
                evaluate_binary_model(
                    models_by_variant[variant_name],
                    dataset.test_data,
                    dataset.test_labels_binary,
                    title,
                )
            )

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


def save_artifacts(
    save_dir: str | Path,
    dataset_key: str,
    multilabel_models: dict[str, models.Model],
    binary_models: dict[str, models.Model],
    multilabel_histories: dict[str, callbacks.History],
    binary_histories: dict[str, callbacks.History],
    multilabel_results: pd.DataFrame,
    binary_results: pd.DataFrame,
) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for variant_name, model in multilabel_models.items():
        model.save(save_dir / f"model_{dataset_key}_{variant_name}.keras")

    for variant_name, model in binary_models.items():
        model.save(save_dir / f"bin_model_{dataset_key}_{variant_name}.keras")

    histories = {
        **{f"history_{dataset_key}_{key}": value.history for key, value in multilabel_histories.items()},
        **{f"bin_history_{dataset_key}_{key}": value.history for key, value in binary_histories.items()},
    }

    with open(save_dir / f"training_histories_{dataset_key}.pkl", "wb") as file_obj:
        import pickle

        pickle.dump(histories, file_obj)

    multilabel_results.to_csv(save_dir / f"uuv_evaluation_results_{dataset_key}.csv", index=False)
    binary_results.to_csv(save_dir / f"binary_uuv_evaluation_results_{dataset_key}.csv", index=False)

    return save_dir


def zip_artifacts(save_dir: str | Path, archive_name: str | Path) -> Path:
    save_dir = Path(save_dir)
    archive_name = Path(archive_name)
    archive_base = archive_name.with_suffix("")
    archive_path = shutil.make_archive(str(archive_base), "zip", save_dir)
    return Path(archive_path)


def extract_zip(zip_path: str | Path, extract_to: str | Path) -> Path:
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    return extract_to
