#!/usr/bin/env python3
"""
Script to add an 'is_question' column to SBSAT question CSV files.
The column is True from the start until the first "1)" (excluding it), and False for the rest.
"""

from pathlib import Path

import pandas as pd

# Hardcoded mappings for files that need manual correction
# Key: filename, Value: index where is_question should become False
HARDCODED_SPLIT_INDICES = {
    'question-flytrap-2_words.csv': 21,
    'question-genome-1_words.csv': 11,
    'question-genome-2_words.csv': 14,
    'question-genome-5_words.csv': 18,
    'question-northpole-1_words.csv': 19,
    'question-northpole-3_words.csv': 15,
}


def add_is_question_column(csv_path):
    """
    Add an 'is_question' column to a CSV file.

    Args:
        csv_path: Path to the CSV file
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Check if this file has a hardcoded split index
    if csv_path.name in HARDCODED_SPLIT_INDICES:
        split_index = HARDCODED_SPLIT_INDICES[csv_path.name]
        df['is_question'] = False
        df.loc[: split_index - 1, 'is_question'] = True
        df.to_csv(csv_path, index=False)
        print(
            f'Updated {csv_path.name} - added is_question column using HARDCODED split at index {split_index}'
        )
        return

    # Find the first occurrence of any answer choice pattern (1), 2), 3), 4), etc.)
    # or a word ending with "?"
    answer_patterns = ['1)', '2)', '3)', '4)']
    first_answer_index = None
    first_pattern = None

    for pattern in answer_patterns:
        pattern_index = df[df['word'] == pattern].index
        if len(pattern_index) > 0:
            if first_answer_index is None or pattern_index[0] < first_answer_index:
                first_answer_index = pattern_index[0]
                first_pattern = pattern

    # Also check for words ending with "?"
    question_mark_indices = df[df['word'].astype(str).str.endswith('?')].index
    if len(question_mark_indices) > 0:
        # The split should be AFTER the last word ending with "?"
        last_question_mark_index = question_mark_indices[-1] + 1
        if first_answer_index is None or last_question_mark_index < first_answer_index:
            first_answer_index = last_question_mark_index
            first_pattern = (
                f"word ending with '?' at index {last_question_mark_index - 1}"
            )

    if first_answer_index is not None:
        # Get the first index where an answer choice appears
        split_index = first_answer_index

        # Create the is_question column
        df['is_question'] = False
        df.loc[: split_index - 1, 'is_question'] = True

        # Save the updated CSV
        df.to_csv(csv_path, index=False)
        print(
            f"Updated {csv_path.name} - added is_question column (True for first {split_index} rows, split at '{first_pattern}')"
        )
    else:
        print(f'Warning: No answer choice pattern found in {csv_path.name}')


def main():
    """Process all question-* CSV files in the SBSAT stimuli directory."""
    stimuli_dir = Path('data/SBSAT/stimuli')

    # Find all question-*_words.csv files
    question_csv_files = sorted(stimuli_dir.glob('question-*_words.csv'))

    print(f'Found {len(question_csv_files)} question CSV files')
    print('=' * 60)

    for csv_path in question_csv_files:
        add_is_question_column(csv_path)

    print('=' * 60)
    print('Done!')


if __name__ == '__main__':
    main()
