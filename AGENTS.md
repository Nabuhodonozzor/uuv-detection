# Repository Notes

- This repository contains notebook-based experiments for UUV sound detection. Each model family trains both a 17-label multilabel classifier and a binary UUV-versus-non-UUV classifier.
- CNN notebooks are in `CNN/`: `CNN_light.ipynb`, `CNN_medium.ipynb`, and `CNN_heavy.ipynb`. `CNN.ipynb` is the original combined CNN notebook.
- RNN notebooks are in `RNN/`: `RNN_mfcc10.ipynb`, `RNN_mfcc20.ipynb`, and `RNN_mfcc40.ipynb`; they train BiLSTMs on the corresponding MFCC feature sets.
- `SVM/` is reserved for SVM-based UUV detection notebooks and is currently empty.
- This repo is Colab-oriented and has no manifests, lockfiles, CI workflows, or automated test/lint commands.
- Shared notebook logic is split into `utils/common_utils.py`, `CNN/cnn_utils.py`, `RNN/rnn_utils.py`, and `SVM/svm_utils.py`; keep common data preparation, evaluation, plotting, and archive helpers in `common_utils.py`, and model-specific logic in its corresponding model directory.
- Dataset-specific notebooks are split by spectrogram dataset: `CNN/CNN_light.ipynb`, `CNN/CNN_medium.ipynb`, and `CNN/CNN_heavy.ipynb`.
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
- If shared behavior changes, update the appropriate utility module first and keep the split notebooks thin.

## Colab Runtime Notes

- Local imports from the utility modules work in Colab only if the required utility files exist in the remote Colab runtime.
- The VS Code Google Colab extension sends notebook code to the remote runtime, but local `.py` files may not automatically be available there depending on extension sync behavior.
- Each split notebook includes a setup cell that checks for its required common and model utility files in the current working directory, `/content`, or `/content/drive/MyDrive/STUDA/src`.
- If a utility file is missing, the setup cell fails clearly and instructs the user to sync it with the VS Code Colab extension, upload it to `/content`, mount Google Drive, or set `UTILS_GITHUB_RAW_BASE_URL`.
- If using GitHub, set `UTILS_GITHUB_RAW_BASE_URL` in the notebook setup cell to the repository raw-file base URL; the notebook will download its required utility files into `/content` before importing.
