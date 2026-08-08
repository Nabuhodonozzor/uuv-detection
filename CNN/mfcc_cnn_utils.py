from keras import callbacks, layers, metrics, models, optimizers

from common_utils import CLASSES, DatasetVariants


def build_mfcc_cnn(input_shape: tuple[int, int], model_type: str = "multilabel", num_classes: int = len(CLASSES)) -> models.Model:
    """Build a temporal CNN for MFCC arrays shaped (time_frames, coefficients)."""
    model = models.Sequential([
        layers.Input(shape=input_shape), layers.LayerNormalization(),
        layers.Conv1D(64, 7, padding="same", use_bias=False, name="conv_1"), layers.BatchNormalization(name="batch_norm_1"), layers.Activation("relu", name="relu_1"), layers.MaxPooling1D(2, name="pool_1"), layers.SpatialDropout1D(0.15, name="spatial_dropout_1"),
        layers.Conv1D(128, 5, padding="same", use_bias=False, name="conv_2"), layers.BatchNormalization(name="batch_norm_2"), layers.Activation("relu", name="relu_2"), layers.MaxPooling1D(2, name="pool_2"), layers.SpatialDropout1D(0.20, name="spatial_dropout_2"),
        layers.Conv1D(256, 3, padding="same", use_bias=False, name="conv_3"), layers.BatchNormalization(name="batch_norm_3"), layers.Activation("relu", name="relu_3"), layers.MaxPooling1D(2, name="pool_3"), layers.SpatialDropout1D(0.30, name="spatial_dropout_3"),
        layers.Conv1D(256, 3, padding="same", use_bias=False, name="conv_4"), layers.BatchNormalization(name="batch_norm_4"), layers.Activation("relu", name="relu_4"), layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation="relu"), layers.Dropout(0.40),
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


def get_mfcc_cnn_callbacks() -> list[callbacks.Callback]:
    return [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]


def build_mfcc_cnn_models_for_variants(variants: DatasetVariants, model_type: str) -> dict[str, models.Model]:
    return {
        "normal": build_mfcc_cnn(variants.normal.train_data.shape[1:], model_type),
        "M": build_mfcc_cnn(variants.m.train_data.shape[1:], model_type),
        "W": build_mfcc_cnn(variants.w.train_data.shape[1:], model_type),
    }
