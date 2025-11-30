#!/usr/bin/env python3
"""
Script to validate that checkpoint files exist for every fold of every model-task combination.
Checks that each model folder has .ckpt files for all expected folds.
Only checks Deep Learning models (TrainerDL) since ML models don't use checkpointing.
Now also skips XGBoostRegressor models and validates checkpoint modification date.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration
BASE_DIR = Path('/data/home/shubi/eyebench_private')
OUTPUTS_DIR = BASE_DIR / 'outputs'
EXPECTED_FOLDS = 4

# Task-specific fold counts
TASK_FOLD_COUNTS = {
    'OneStop_RC': 10,
    # All other tasks default to 4 folds
}


# ANSI color codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color


def check_folder_for_checkpoints(
    folder_path: Path, task_name: str, expected_folds: int, min_date: datetime
) -> Tuple[bool, List[str], Dict[str, bool]]:
    """
    Check if a model folder has checkpoint files for the given task and after a given date.

    Returns:
        Tuple of (is_valid, list_of_issues, fold_status_dict)
    """
    issues = []
    fold_status = {}

    for i in range(expected_folds):
        fold_dir = folder_path / f'fold_index={i}'
        fold_status[f'fold_{i}'] = False

        if not fold_dir.is_dir():
            issues.append(f'Missing fold_index={i} directory')
            continue

        ckpt_files = list(fold_dir.glob('*.ckpt'))
        if not ckpt_files:
            issues.append(
                f'fold_index={i}: Missing checkpoint file (*.ckpt) in {fold_dir}'
            )
            continue

        valid_ckpt_found = False
        for ckpt in ckpt_files:
            mod_time = datetime.fromtimestamp(ckpt.stat().st_mtime)
            date_str = mod_time.strftime('%Y-%m-%d %H:%M')
            if mod_time >= min_date:
                print(
                    f'  {Colors.GREEN}✓{Colors.NC} fold_index={i} → {ckpt.name} ({date_str}) ✅ valid'
                )
                valid_ckpt_found = True
            else:
                print(
                    f'  {Colors.RED}✗{Colors.NC} fold_index={i} → {ckpt.name} ({date_str}) ❌ too old'
                )

        if valid_ckpt_found:
            fold_status[f'fold_{i}'] = True
        else:
            issues.append(
                f'fold_index={i}: All checkpoint files are older than {min_date.strftime("%Y-%m-%d")}'
            )

    return len(issues) == 0, issues, fold_status


def extract_model_name(folder_name: str) -> str:
    """Extract the model name from the folder name."""
    parts = folder_name.split(',')
    for part in parts:
        if '+model=' in part:
            return part.split('=')[1]
    return 'Unknown'


def extract_task_name(folder_name: str) -> str:
    """Extract the task name from the folder name."""
    parts = folder_name.split(',')
    for part in parts:
        if '+data=' in part:
            return part.split('=')[1]
    return 'Unknown'


def get_expected_folds(task_name: str, default_folds: int) -> int:
    """Get the expected number of folds for a given task."""
    return TASK_FOLD_COUNTS.get(task_name, default_folds)


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description='Validate that checkpoint files exist for every fold of every model-task combination.'
    )
    parser.add_argument(
        'task',
        type=str,
        nargs='?',
        help='Task name to check (e.g., CopCo_RCS, OneStop_RC, MECOL2_LEX). If not provided, checks all tasks.',
    )
    parser.add_argument(
        '--folds',
        type=int,
        default=None,
        help=f'Number of expected folds (default: task-specific or {EXPECTED_FOLDS})',
    )
    parser.add_argument(
        '--date-after',
        type=str,
        default='2025-10-22',
        help='Only consider checkpoint files modified after this date (format: YYYY-MM-DD)',
    )
    parser.add_argument(
        '--show-all',
        action='store_true',
        help='Show all models even if valid (default: only show issues)',
    )
    args = parser.parse_args()

    task_name = args.task
    if args.folds is not None:
        expected_folds_default = args.folds
    else:
        expected_folds_default = EXPECTED_FOLDS

    # Parse date
    try:
        min_date = datetime.strptime(args.date_after, '%Y-%m-%d')
    except ValueError:
        print(
            f'{Colors.RED}✗ Invalid date format: {args.date_after} (expected YYYY-MM-DD){Colors.NC}'
        )
        return 1

    print('=' * 80)
    print('Checkpoint File Validation - outputs')
    print('=' * 80)
    print(f'Directory: {OUTPUTS_DIR}')
    if task_name:
        print(f'Task filter: {task_name}')
        expected_folds_for_task = get_expected_folds(task_name, expected_folds_default)
        print(f'Expected folds: {expected_folds_for_task}')
    else:
        print('Task filter: ALL TASKS')
        print(
            f'Expected folds: task-specific (OneStop_RC: 10, others: {expected_folds_default})'
        )
    print(
        f'Valid checkpoint date range: {Colors.CYAN}after {min_date.strftime("%Y-%m-%d")}{Colors.NC}'
    )
    print('=' * 80)
    print()

    if not OUTPUTS_DIR.exists():
        print(f'{Colors.RED}✗ Directory does not exist: {OUTPUTS_DIR}{Colors.NC}')
        return 1

    # Find folders
    if task_name:
        task_pattern = f'+data={task_name},'
        matching_folders = sorted(
            [
                d
                for d in OUTPUTS_DIR.iterdir()
                if d.is_dir()
                and task_pattern in d.name
                and '+trainer=TrainerDL' in d.name
                and 'DEBUG' not in d.name.lower()
            ]
        )
    else:
        matching_folders = sorted(
            [
                d
                for d in OUTPUTS_DIR.iterdir()
                if d.is_dir()
                and '+data=' in d.name
                and '+trainer=TrainerDL' in d.name
                and 'DEBUG' not in d.name.lower()
            ]
        )

    if not matching_folders:
        msg = (
            f'No matching folders found for {task_name}'
            if task_name
            else 'No TrainerDL folders found.'
        )
        print(f'{Colors.YELLOW}! {msg}{Colors.NC}')
        return 0

    # Group by task
    task_folders = {}
    for folder in matching_folders:
        task = extract_task_name(folder.name)
        task_folders.setdefault(task, []).append(folder)

    total_models = 0
    valid_models = 0
    invalid_models = 0
    partial_models = 0
    skipped_models = 0
    models_with_issues = []

    for task, folders in sorted(task_folders.items()):
        if task_name and task != task_name:
            continue

        task_expected_folds = get_expected_folds(task, expected_folds_default)

        print(
            f'{Colors.BLUE}Task: {task}{Colors.NC} (expected folds: {task_expected_folds})'
        )
        print('-' * 80)

        for folder in sorted(folders):
            total_models += 1
            model_name = extract_model_name(folder.name)

            # Skip XGBoostRegressor
            if model_name == 'XGBoostRegressor':
                skipped_models += 1
                print(
                    f'{Colors.CYAN}⏩ Skipping {model_name} (no checkpoints expected){Colors.NC}'
                )
                continue

            is_valid, issues, fold_status = check_folder_for_checkpoints(
                folder, task, task_expected_folds, min_date
            )
            valid_folds = sum(1 for v in fold_status.values() if v)

            if is_valid:
                valid_models += 1
                if args.show_all:
                    print(
                        f'{Colors.GREEN}✓ {model_name}{Colors.NC} (all {task_expected_folds} folds valid)'
                    )
            elif valid_folds > 0:
                partial_models += 1
                models_with_issues.append(f'{task}/{model_name}')
                print(
                    f'{Colors.YELLOW}⚠ {model_name}{Colors.NC} ({valid_folds}/{task_expected_folds} folds valid)'
                )
                for issue in issues:
                    print(f'  {Colors.YELLOW}! {issue}{Colors.NC}')
            else:
                invalid_models += 1
                models_with_issues.append(f'{task}/{model_name}')
                print(f'{Colors.RED}✗ {model_name}{Colors.NC} (no valid folds)')
                for issue in issues:
                    print(f'  {Colors.RED}✗ {issue}{Colors.NC}')
            print()

        print()

    # Summary
    print('=' * 80)
    print('Summary')
    print('=' * 80)
    print(f'Total models checked: {total_models}')
    print(f'Valid models: {Colors.GREEN}{valid_models}{Colors.NC}')
    print(f'Partial models: {Colors.YELLOW}{partial_models}{Colors.NC}')
    print(f'Invalid models: {Colors.RED}{invalid_models}{Colors.NC}')
    print(f'Skipped models: {Colors.CYAN}{skipped_models}{Colors.NC}')
    print(
        f'Valid checkpoint date: {Colors.CYAN}after {min_date.strftime("%Y-%m-%d")}{Colors.NC}'
    )

    if models_with_issues:
        print(f'\nModels with issues ({len(models_with_issues)}):')
        for m in models_with_issues:
            print(f'  - {m}')
        print()

    if invalid_models == 0 and partial_models == 0:
        print(
            f'{Colors.GREEN}✓ All models have complete and up-to-date checkpoint files!{Colors.NC}'
        )
        return 0
    elif invalid_models == 0:
        print(f'{Colors.YELLOW}⚠ Some models have partial checkpoint files.{Colors.NC}')
        return 1
    else:
        print(
            f'{Colors.RED}✗ Some models have missing or outdated checkpoint files.{Colors.NC}'
        )
        return 1


if __name__ == '__main__':
    # exit(main())

    # if __name__ == '__main__':
    #     import sys
    #     import traceback

    #     try:
    #         print('=' * 80)
    #         print(f"{Colors.BLUE}[INFO]{Colors.NC} Starting checkpoint validation script")
    #         print('=' * 80)
    #         print(f"{Colors.CYAN}Command-line arguments:{Colors.NC}")
    #         print(f"  Task:         {sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'ALL'}")
    #         print(f"  --folds:      {next((sys.argv[i+1] for i,a in enumerate(sys.argv) if a=='--folds'), 'default')}")
    #         print(f"  --date-after: {next((sys.argv[i+1] for i,a in enumerate(sys.argv) if a=='--date-after'), '2025-10-22')}")
    #         print(f"  --show-all:   {'--show-all' in sys.argv}")
    #         print('=' * 80)
    #         print()

    #         exit_code = main()
    #         sys.exit(exit_code)

    #     except Exception as e:
    #         print(f"{Colors.RED}[ERROR]{Colors.NC} Unexpected failure: {e}")
    #         traceback.print_exc()
    #         sys.exit(1)

    import sys
    import traceback
    from datetime import datetime

    # --- Logging setup ---
    log_dir = Path(BASE_DIR) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'validate_checkpoints.log'

    class TeeLogger:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    log_fh = open(log_file, 'w', encoding='utf-8')
    sys.stdout = TeeLogger(sys.stdout, log_fh)
    sys.stderr = TeeLogger(sys.stderr, log_fh)

    print('=' * 80)
    print(f'{Colors.BLUE}[INFO]{Colors.NC} Starting checkpoint validation script')
    print(f'{Colors.CYAN}Logs will also be saved to:{Colors.NC} {log_file}')
    print('=' * 80)
    print()

    try:
        # --- Force show_all to False by default ---
        if '--show-all' not in sys.argv:
            sys.argv = [arg for arg in sys.argv if arg != '--show-all']
            print(
                f'{Colors.YELLOW}[INFO]{Colors.NC} Forcing show_all = False (flag not passed)'
            )
        else:
            print(f'{Colors.CYAN}[INFO]{Colors.NC} show_all = True (flag detected)')

        # Run main
        exit_code = main()

        print()
        print(
            f'{Colors.BLUE}[INFO]{Colors.NC} Script finished with exit code {exit_code}'
        )
        print(f'{Colors.CYAN}[INFO]{Colors.NC} Full log written to {log_file}')
        sys.exit(exit_code)

    except Exception as e:
        print(f'{Colors.RED}[ERROR]{Colors.NC} Unexpected failure: {e}')
        traceback.print_exc()
        sys.exit(1)
    finally:
        log_fh.close()
