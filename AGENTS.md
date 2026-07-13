# Repository Notes

- This repo is Colab-oriented and has no manifests, lockfiles, CI workflows, or automated test/lint commands.
- Shared notebook logic lives in `spectrogram_pipeline.py`; keep data preparation, model builders, callbacks, evaluation helpers, plotting helpers, and artifact-saving helpers there instead of duplicating them across notebooks.
- Dataset-specific notebooks are split by spectrogram dataset: `CNN_light.ipynb`, `CNN_medium.ipynb`, and `CNN_heavy.ipynb`.
- `CNN.ipynb` is the original large notebook. Prefer updating the split notebooks and shared module for new work.
- Treat notebook cell execution as the verification path. Prefer running only relevant cells, because each split notebook can still train 3 multilabel CNNs plus 3 binary CNNs at up to 50 epochs each.
- The split notebooks use shell `!kaggle` commands, `google.colab.userdata`, and `google.colab.files`; they expect Kaggle credentials from a Colab secret named `Kaggle` written to `/root/.kaggle/access_token`.
- Data is downloaded from Kaggle into `/content`: `pawedyrda/mel-spectrogram-light` (~630 MB), `pawedyrda/mel-spectrogram-medium` (~2.41 GB), or `pawedyrda/mel-spectrogram-heavy` (~9.54 GB), then extracted under the matching `/content/mel-spectrogram-*` directory.
- Each split notebook must download and process only its own dataset variant.
- Required runtime packages are notebook/script imports only: `numpy`, `scikit-learn`, `keras`, `pandas`, `matplotlib`, `IPython`, `kaggle`, `kagglehub`, and Colab APIs. No repo-local dependency file exists.
- Keep generated files out of the repo unless explicitly requested: `saved_artifacts/`, `saved_models_and_results*.zip`, Kaggle dataset metadata, downloaded ZIPs, extracted spectrogram data, and trained `.keras` models are runtime artifacts.
- Model outputs use sigmoid + `binary_crossentropy` for both multilabel classification and binary UUV detection; evaluation thresholds predictions at `0.5`.
- The fixed class list has 17 labels and `UUV` is reused for binary labels via `DatasetSplits.train_labels_binary`, `val_labels_binary`, and `test_labels_binary`; keep class ordering synchronized across light/medium/heavy variants.

## Notebook Split Workflow

- Use `prepare_dataset_variants(data_path)` from `spectrogram_pipeline.py` to prepare normal, M-filtered, and W-filtered splits together.
- Each split notebook should call `prepare_dataset_variants(DATA_PATH)` once and then train/evaluate both multilabel CNNs and binary UUV CNNs for the normal, M, and W splits.
- Keep notebooks orchestration-focused: runtime setup, dataset download/extraction, import shared helpers, prepare variants, train, evaluate, save artifacts.
- Keep notebooks checked in with outputs cleared to avoid large `.ipynb` files.
- If shared behavior changes, update `spectrogram_pipeline.py` first and keep the three split notebooks thin.

## Colab Runtime Notes

- Local imports like `from spectrogram_pipeline import ...` work in Colab only if `spectrogram_pipeline.py` exists in the remote Colab runtime.
- The VS Code Google Colab extension sends notebook code to the remote runtime, but local `.py` files may not automatically be available there depending on extension sync behavior.
- Each split notebook includes a setup cell that checks for `spectrogram_pipeline.py` in the current working directory, `/content`, or `/content/drive/MyDrive/STUDA/src`.
- If the module is missing, the setup cell fails clearly and instructs the user to sync it with the VS Code Colab extension, upload it to `/content`, mount Google Drive, or set `PIPELINE_GITHUB_RAW_URL`.
- If using GitHub, set `PIPELINE_GITHUB_RAW_URL` in the notebook setup cell to a raw GitHub URL for `spectrogram_pipeline.py`; the notebook will download it into `/content` before importing.
