from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from common_utils import DatasetVariants, prepare_mfccs


def flatten_mfcc_features(mfccs: np.ndarray) -> np.ndarray:
    mfccs = prepare_mfccs(mfccs)
    return mfccs.reshape(mfccs.shape[0], -1)


def build_svm_models_for_variants(variants: DatasetVariants, model_type: str) -> dict[str, Pipeline]:
    if model_type not in {"multilabel", "binary"}:
        raise ValueError("model_type must be either 'multilabel' or 'binary'")

    def build_model() -> Pipeline:
        classifier = LinearSVC(C=1.0, class_weight="balanced", dual=True, max_iter=10_000)
        if model_type == "multilabel":
            classifier = OneVsRestClassifier(classifier)
        return Pipeline([("flatten", FunctionTransformer(flatten_mfcc_features, validate=False)), ("scale", StandardScaler()), ("classifier", classifier)])

    return {variant_name: build_model() for variant_name in ("normal", "M", "W")}


def train_svm_models_for_variants(models_by_variant: dict[str, Pipeline], variants: DatasetVariants, model_type: str) -> dict[str, Pipeline]:
    for variant_name, dataset in {"normal": variants.normal, "M": variants.m, "W": variants.w}.items():
        train_labels = dataset.train_labels_multi if model_type == "multilabel" else dataset.train_labels_binary
        print(f"Training {model_type} SVM for variant: {variant_name}")
        models_by_variant[variant_name].fit(dataset.train_data, train_labels)
    return models_by_variant


def save_svm_artifacts(save_dir: str | Path, dataset_key: str, multilabel_models: dict[str, Pipeline], binary_models: dict[str, Pipeline], multilabel_results: pd.DataFrame, binary_results: pd.DataFrame) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for variant_name, model in multilabel_models.items():
        joblib.dump(model, save_dir / f"svm_model_{dataset_key}_{variant_name}.joblib")
    for variant_name, model in binary_models.items():
        joblib.dump(model, save_dir / f"svm_bin_model_{dataset_key}_{variant_name}.joblib")
    multilabel_results.to_csv(save_dir / f"svm_uuv_evaluation_results_{dataset_key}.csv", index=False)
    binary_results.to_csv(save_dir / f"svm_binary_uuv_evaluation_results_{dataset_key}.csv", index=False)
    return save_dir
