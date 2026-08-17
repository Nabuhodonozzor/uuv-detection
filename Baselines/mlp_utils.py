from keras import callbacks, layers, metrics, models, optimizers

from common_utils import CLASSES, DatasetVariants


def build_mfcc_mlp(input_shape: tuple[int, int], model_type: str = "multilabel", num_classes: int = len(CLASSES)) -> models.Model:
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LayerNormalization(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
    ])
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
            metrics.AUC(name="roc_auc", curve="ROC", multi_label=model_type == "multilabel"),
            metrics.AUC(name="pr_auc", curve="PR", multi_label=model_type == "multilabel"),
            metrics.Precision(name="precision", thresholds=0.5),
            metrics.Recall(name="recall", thresholds=0.5),
        ],
    )
    return model


def get_mlp_callbacks() -> list[callbacks.Callback]:
    return [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ]


def build_mlp_models_for_variants(variants: DatasetVariants, model_type: str) -> dict[str, models.Model]:
    return {
        "normal": build_mfcc_mlp(variants.normal.train_data.shape[1:], model_type),
        "M": build_mfcc_mlp(variants.m.train_data.shape[1:], model_type),
        "W": build_mfcc_mlp(variants.w.train_data.shape[1:], model_type),
    }
