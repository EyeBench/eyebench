set -euxo pipefail
DATASET_LIST="${1:-}"

if [ -n "$DATASET_LIST" ]; then
  python src/data/preprocessing/download_data.py --dataset "$DATASET_LIST"
  python src/data/preprocessing/union_raw_files.py --dataset "$DATASET_LIST"
  python src/data/preprocessing/preprocess_data.py --dataset "$DATASET_LIST"
  python src/data/preprocessing/create_folds.py --dataset "$DATASET_LIST" --do_not_recreate_trial_folds --do_not_recreate_item_subject_folds
  python src/data/preprocessing/stats.py --dataset "$DATASET_LIST"
else
  # No datasets passed — run without dataset argument
  python src/data/preprocessing/download_data.py
  python src/data/preprocessing/union_raw_files.py
  python src/data/preprocessing/preprocess_data.py
  python src/data/preprocessing/create_folds.py --do_not_recreate_trial_folds --do_not_recreate_item_subject_folds
  python src/data/preprocessing/stats.py
fi

