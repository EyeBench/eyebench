import os

import joblib
import pandas as pd
from tqdm import tqdm

BASE_DIR = 'data/old_cache/features'
TASKS = [
    'CopCo_RCS',
    'CopCo_TYP',
    'IITBHGC_CV',
    'MECOL2_LEX',
    'OneStop_RC',
    'PoTeC_DE',
    'PoTeC_RC',
    'SBSAT_STD',
    'SBSAT_RC',
]
MODEL = 'AhnCNN'

TRAIN_FILE = 'train_train.pkl'
VAL_TEST_FILES = [
    'seen_subject_unseen_item_val.pkl',
    'seen_subject_unseen_item_test.pkl',
    'unseen_subject_unseen_item_val.pkl',
    'unseen_subject_unseen_item_test.pkl',
]

results = {}

# Outer loop over tasks with tqdm
for task in TASKS:
    if task == 'OneStop_RC':
        folds = range(10)
    else:
        folds = range(4)

    results[task] = {}

    # Inner loop over folds with tqdm
    for fold_idx in tqdm(folds, desc=f'Processing {task}', unit='fold'):
        FOLD = f'fold_{fold_idx}'
        fold_data = {}

        # Load train
        train_path = os.path.join(BASE_DIR, f'{task}_{MODEL}', FOLD, TRAIN_FILE)
        train_data = joblib.load(train_path)
        train_df = train_data[4].obj
        train_df = train_df.drop_duplicates(subset=['unique_trial_id'])
        fold_data['train'] = len(train_df)

        # Load validation/test and drop duplicates
        for file_name in VAL_TEST_FILES:
            file_path = os.path.join(BASE_DIR, f'{task}_{MODEL}', FOLD, file_name)
            key_name = file_name.replace('.pkl', '')
            if os.path.exists(file_path):
                df = joblib.load(file_path)
                if isinstance(df, tuple):
                    df = df[4].obj
                df = df.drop_duplicates(subset=['unique_trial_id'])
                fold_data[key_name] = len(df)
            else:
                fold_data[key_name] = None

        results[task][FOLD] = fold_data

# Convert results to DataFrame
rows = []
for task, folds_dict in results.items():
    for fold, counts in folds_dict.items():
        row = {'task': task, 'fold': fold}
        row.update(counts)
        rows.append(row)

df_final = pd.DataFrame(rows)

# Reorder columns
columns_order = ['task', 'fold', 'train'] + [
    f.replace('.pkl', '') for f in VAL_TEST_FILES
]
df_final = df_final[columns_order]

# Export CSV
df_final.to_csv('sample_values.csv', index=False)
print('\nExported sample_values.csv successfully!')
