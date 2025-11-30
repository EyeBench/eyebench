#!/usr/bin/env python3
"""
Remove duplicates from PoTeC_DE trial_level_test_results.csv files.
Deduplicates based on: eval_regime, eval_type, fold_index, unique_paragraph_id, participant_id
"""

import glob

import pandas as pd

# Find all PoTeC_DE trial_level_test_results.csv files
pattern = 'results/raw/*PoTeC_DE*/*/trial_level_test_results.csv'
csv_files = glob.glob(pattern)

print(f'Found {len(csv_files)} CSV files to process')

# Columns to use for identifying duplicates
dedup_columns = [
    'eval_regime',
    'eval_type',
    'fold_index',
    'unique_paragraph_id',
    'participant_id',
]

for csv_file in csv_files:
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        original_count = len(df)

        # Remove duplicates based on the specified columns
        # Keep the first occurrence
        df_deduped = df.drop_duplicates(subset=dedup_columns, keep='first')

        deduped_count = len(df_deduped)
        duplicates_removed = original_count - deduped_count

        if duplicates_removed > 0:
            # Save the deduplicated data back to the file
            df_deduped.to_csv(csv_file, index=False)
            print(f'✓ {csv_file}')
            print(
                f'  Removed {duplicates_removed} duplicates ({original_count} -> {deduped_count} rows)'
            )
        else:
            print(f'✓ {csv_file} - No duplicates found')

    except Exception as e:
        print(f'✗ Error processing {csv_file}: {e}')

print('\nDeduplication complete!')
