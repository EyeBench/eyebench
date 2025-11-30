#!/usr/bin/env python3
"""
Script to validate cache folder structure.
Checks that each matching folder has 4 folds with 7 files each.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

# Configuration
CACHE_DIR = Path('data/cache/features')
EXPECTED_FOLDS = 4
EXPECTED_FILES = 7

# Expected file names in each fold
EXPECTED_FILE_NAMES = [
    'seen_subject_unseen_item_test.pkl',
    'seen_subject_unseen_item_val.pkl',
    'train_train.pkl',
    'unseen_subject_seen_item_test.pkl',
    'unseen_subject_seen_item_val.pkl',
    'unseen_subject_unseen_item_test.pkl',
    'unseen_subject_unseen_item_val.pkl',
]


# ANSI color codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    NC = '\033[0m'  # No Color


def check_folder(folder_path: Path) -> Tuple[bool, List[str]]:
    """
    Check if a single CopCo_RCS folder has valid structure.

    Args:
        folder_path: Path to the CopCo_RCS_* folder

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check if all 4 folds exist
    fold_count = 0
    for i in range(EXPECTED_FOLDS):
        fold_dir = folder_path / f'fold_{i}'
        if fold_dir.is_dir():
            fold_count += 1
        else:
            issues.append(f'Missing fold_{i}')

    if fold_count != EXPECTED_FOLDS:
        issues.append(f'Expected {EXPECTED_FOLDS} folds, found {fold_count}')

    # Check each fold for the 7 expected files
    for i in range(EXPECTED_FOLDS):
        fold_dir = folder_path / f'fold_{i}'
        if not fold_dir.is_dir():
            continue

        # Count files in fold
        files = [f for f in fold_dir.iterdir() if f.is_file()]
        file_count = len(files)

        if file_count != EXPECTED_FILES:
            issues.append(
                f'fold_{i}: Expected {EXPECTED_FILES} files, found {file_count}'
            )

        # Check for each expected file
        for expected_file in EXPECTED_FILE_NAMES:
            file_path = fold_dir / expected_file
            if not file_path.is_file():
                issues.append(f'fold_{i}: Missing file {expected_file}')

    return len(issues) == 0, issues


def main():
    """Main validation function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Validate cache folder structure for eye-tracking datasets.'
    )
    parser.add_argument(
        'pattern',
        type=str,
        nargs='?',
        default='CopCo_RCS',
        help='Folder name pattern to match (default: CopCo_RCS)',
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=str(CACHE_DIR),
        help=f'Cache directory path (default: {CACHE_DIR})',
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    pattern = args.pattern

    print('=' * 50)
    print(f'{pattern} Cache Structure Validation')
    print('=' * 50)
    print()

    if not cache_dir.exists():
        print(f'{Colors.RED}✗ Cache directory does not exist: {cache_dir}{Colors.NC}')
        return 1

    # Find all matching folders
    matching_folders = sorted(
        [
            d
            for d in cache_dir.iterdir()
            if d.is_dir() and d.name.startswith(f'{pattern}_')
        ]
    )

    if not matching_folders:
        print(
            f'{Colors.YELLOW}! No {pattern}_* folders found in {cache_dir}{Colors.NC}'
        )
        return 0

    # Counter for summary
    total_folders = len(matching_folders)
    valid_folders = 0
    invalid_folders = 0

    # Check each folder
    for folder in matching_folders:
        folder_name = folder.name
        print(f'Checking: {folder_name}')

        is_valid, issues = check_folder(folder)

        if is_valid:
            print(f'  {Colors.GREEN}✓{Colors.NC} All checks passed')
            valid_folders += 1
        else:
            for issue in issues:
                print(f'  {Colors.RED}✗{Colors.NC} {issue}')
            invalid_folders += 1

        print()

    # Final summary
    print('=' * 50)
    print('Summary')
    print('=' * 50)
    print(f'Total {pattern} folders checked: {total_folders}')
    print(f'Valid folders: {Colors.GREEN}{valid_folders}{Colors.NC}')
    print(f'Invalid folders: {Colors.RED}{invalid_folders}{Colors.NC}')
    print()

    if invalid_folders == 0:
        print(f'{Colors.GREEN}✓ All {pattern} cache folders are valid!{Colors.NC}')
        return 0
    else:
        print(f'{Colors.RED}✗ Some {pattern} cache folders have issues.{Colors.NC}')
        return 1


if __name__ == '__main__':
    exit(main())
