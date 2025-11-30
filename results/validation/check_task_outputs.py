#!/usr/bin/env python3
"""
Script to validate that outputs/results exist for a given task.
Checks that each model folder has results for all expected folds with CSV output files.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration
BASE_DIR = Path('/data/home/shubi/eyebench_private')
OUTPUTS_DIR = BASE_DIR / 'outputs'
RESULTS_DIR = BASE_DIR / 'results' / 'raw'
EXPECTED_FOLDS = 4


# ANSI color codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def check_folder_for_task(
    folder_path: Path, task_name: str, expected_folds: int
) -> Tuple[bool, List[str], Dict[str, bool]]:
    """
    Check if a model folder has valid structure for the given task.

    Args:
        folder_path: Path to the model output folder
        task_name: Name of the task (e.g., CopCo_RCS)
        expected_folds: Number of expected folds

    Returns:
        Tuple of (is_valid, list_of_issues, fold_status_dict)
    """
    issues = []
    fold_status = {}

    # Check each expected fold
    for i in range(expected_folds):
        fold_dir = folder_path / f'fold_index={i}'
        fold_status[f'fold_{i}'] = False

        if not fold_dir.is_dir():
            issues.append(f'Missing fold_index={i} directory')
            continue

        # Check for CSV file (trial_level_test_results.csv)
        csv_file = fold_dir / 'trial_level_test_results.csv'

        if csv_file.is_file():
            # Check if file has content (more than just header)
            try:
                file_size = csv_file.stat().st_size
                if file_size > 100:  # Arbitrary small size to ensure it's not empty
                    fold_status[f'fold_{i}'] = True
                else:
                    issues.append(
                        f'fold_index={i}: CSV file is too small (likely empty)'
                    )
            except Exception as e:
                issues.append(f'fold_index={i}: Error checking CSV file - {e}')
        else:
            issues.append(f'fold_index={i}: Missing trial_level_test_results.csv')

    return len(issues) == 0, issues, fold_status


def extract_model_name(folder_name: str) -> str:
    """Extract the model name from the folder name."""
    # Format: +data=TASK,+model=MODEL,+trainer=TRAINER,...
    parts = folder_name.split(',')
    for part in parts:
        if '+model=' in part:
            return part.split('=')[1]
    return 'Unknown'


def main():
    """Main validation function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Validate that outputs/results exist for a given task.'
    )
    parser.add_argument(
        'task',
        type=str,
        help='Task name to check (e.g., CopCo_RCS, OneStop_RC, MECOL2_LEX)',
    )
    parser.add_argument(
        '--folds',
        type=int,
        default=EXPECTED_FOLDS,
        help=f'Number of expected folds (default: {EXPECTED_FOLDS})',
    )
    parser.add_argument(
        '--check-outputs',
        action='store_true',
        help='Check outputs directory (default: checks results/raw)',
    )
    parser.add_argument(
        '--show-all',
        action='store_true',
        help='Show all models even if valid (default: only show issues)',
    )
    args = parser.parse_args()

    task_name = args.task
    expected_folds = args.folds
    check_dir = OUTPUTS_DIR if args.check_outputs else RESULTS_DIR
    dir_type = 'outputs' if args.check_outputs else 'results/raw'

    print('=' * 70)
    print(f'{task_name} Output Validation - {dir_type}')
    print('=' * 70)
    print(f'Directory: {check_dir}')
    print(f'Expected folds: {expected_folds}')
    print('=' * 70)
    print()

    if not check_dir.exists():
        print(f'{Colors.RED}✗ Directory does not exist: {check_dir}{Colors.NC}')
        return 1

    # Find all folders matching the task
    task_pattern = f'+data={task_name},'
    matching_folders = sorted(
        [
            d
            for d in check_dir.iterdir()
            if d.is_dir() and task_pattern in d.name and 'DEBUG' not in d.name
        ]
    )

    if not matching_folders:
        print(f'{Colors.YELLOW}! No folders found for task: {task_name}{Colors.NC}')
        print(f'  Pattern searched: {task_pattern}')
        return 0

    # Counter for summary
    total_models = len(matching_folders)
    valid_models = 0
    invalid_models = 0
    partial_models = 0

    # Track models with issues
    models_with_issues = []

    # Check each folder
    for folder in matching_folders:
        folder_name = folder.name
        model_name = extract_model_name(folder_name)

        is_valid, issues, fold_status = check_folder_for_task(
            folder, task_name, expected_folds
        )

        # Count how many folds are valid
        valid_folds = sum(1 for v in fold_status.values() if v)

        if is_valid:
            valid_models += 1
            if args.show_all:
                print(f'{Colors.GREEN}✓{Colors.NC} {model_name}')
                print(f'  All {expected_folds} folds have valid CSV outputs')
                print()
        elif valid_folds > 0:
            partial_models += 1
            models_with_issues.append(model_name)
            print(
                f'{Colors.YELLOW}⚠{Colors.NC} {model_name} (partial: {valid_folds}/{expected_folds} folds)'
            )
            for issue in issues:
                print(f'  {Colors.YELLOW}!{Colors.NC} {issue}')
            print()
        else:
            invalid_models += 1
            models_with_issues.append(model_name)
            print(f'{Colors.RED}✗{Colors.NC} {model_name} (no valid folds)')
            for issue in issues:
                print(f'  {Colors.RED}✗{Colors.NC} {issue}')
            print()

    # Final summary
    print('=' * 70)
    print('Summary')
    print('=' * 70)
    print(f'Task: {task_name}')
    print(f'Total models checked: {total_models}')
    print(f'Valid models (all folds): {Colors.GREEN}{valid_models}{Colors.NC}')
    print(f'Partial models (some folds): {Colors.YELLOW}{partial_models}{Colors.NC}')
    print(f'Invalid models (no folds): {Colors.RED}{invalid_models}{Colors.NC}')

    if models_with_issues:
        print(f'\nModels with issues: {", ".join(models_with_issues)}')

    print()

    if invalid_models == 0 and partial_models == 0:
        print(
            f'{Colors.GREEN}✓ All models for {task_name} have complete results!{Colors.NC}'
        )
        return 0
    elif invalid_models == 0:
        print(f'{Colors.YELLOW}⚠ Some models have partial results.{Colors.NC}')
        return 1
    else:
        print(
            f'{Colors.RED}✗ Some models have missing or incomplete results.{Colors.NC}'
        )
        return 1


if __name__ == '__main__':
    exit(main())
