"""
Script to test loading all .pkl files in data/cache/features directory.
Uses joblib to load each pickle file and reports success or errors.
"""

import argparse
import signal
from pathlib import Path
from typing import List, Tuple

import joblib


class TimeoutError(Exception):
    """Custom timeout exception."""

    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError('Loading operation timed out')


def find_pkl_files(root_dir: str) -> List[Path]:
    """
    Recursively find all .pkl files in the given directory.

    Args:
        root_dir: Root directory to search

    Returns:
        List of Path objects for all .pkl files found
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f'Directory does not exist: {root_dir}')
        return []

    pkl_files = list(root_path.rglob('*.pkl'))
    return pkl_files


def load_pkl_file(file_path: Path, timeout: int = 5) -> Tuple[bool, str, object]:
    """
    Try to load a pickle file using joblib with a timeout.

    Args:
        file_path: Path to the .pkl file
        timeout: Maximum time in seconds to wait for loading (default: 30)

    Returns:
        Tuple of (success, message, data)
    """
    # Set up timeout alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        data = joblib.load(file_path)
        signal.alarm(0)  # Cancel the alarm
        return True, 'Success', data
    except TimeoutError:
        signal.alarm(0)  # Cancel the alarm
        return False, f'Timeout after {timeout}s', None
    except KeyboardInterrupt:
        signal.alarm(0)  # Cancel the alarm
        raise  # Re-raise to allow script interruption
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm
        return False, str(e), None


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Load and test all .pkl files in specified directories'
    )
    parser.add_argument(
        'pattern',
        nargs='?',
        default='MECOL2_LEX_AhnCNN*',
        help='Directory pattern to search (default: SBSAT_RC*)',
    )
    parser.add_argument(
        '--base-dir',
        default='data/cache/features',
        help='Base directory to search in (default: data/cache/features)',
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=5,
        help='Timeout in seconds for loading each file (default: 30)',
    )
    args = parser.parse_args()

    # Directory to search
    cache_base = args.base_dir
    pattern = args.pattern

    # Find all matching directories
    base_path = Path(cache_base)
    if not base_path.exists():
        print(f'Directory does not exist: {cache_base}')
        return

    matching_dirs = list(base_path.glob(pattern))

    if not matching_dirs:
        print(f"No directories matching '{pattern}' found in {cache_base}")
        return

    print(f"Found {len(matching_dirs)} directories matching '{pattern}':")
    for d in matching_dirs:
        print(f'  - {d.name}')
    print('=' * 80)

    # Find all .pkl files in all matching directories
    all_pkl_files = []
    for matching_dir in matching_dirs:
        pkl_files = find_pkl_files(str(matching_dir))
        all_pkl_files.extend(pkl_files)

    if not all_pkl_files:
        print(f"No .pkl files found in directories matching '{pattern}'")
        return

    print(f'Found {len(all_pkl_files)} .pkl files\n')

    # Try to load each file
    success_count = 0
    failed_count = 0
    results = []

    for i, pkl_file in enumerate(all_pkl_files, 1):
        relative_path = pkl_file.relative_to(cache_base)
        print(f'[{i}/{len(all_pkl_files)}] Loading: {relative_path}')
        success, message, data = load_pkl_file(pkl_file, timeout=args.timeout)

        if success:
            success_count += 1
            # Get some info about the loaded data
            data_type = type(data).__name__
            data_info = ''

            if hasattr(data, 'shape'):
                data_info = f' (shape: {data.shape})'
            elif hasattr(data, '__len__'):
                try:
                    data_info = f' (length: {len(data)})'
                except Exception as e:
                    print(f'Error getting length of data: {e}')

            result = f'✓ {relative_path} - {data_type}{data_info}'
            print(result)
            results.append((True, relative_path, result))
        else:
            failed_count += 1
            result = f'✗ {relative_path} - ERROR: {message}'
            print(result)
            results.append((False, relative_path, result))

    # Summary
    print('\n' + '=' * 80)
    print('SUMMARY:')
    print(f'  Total files: {len(all_pkl_files)}')
    print(f'  Successfully loaded: {success_count}')
    print(f'  Failed to load: {failed_count}')

    if failed_count > 0:
        print('\nFailed files:')
        for success, path, result in results:
            if not success:
                print(f'  - {path}')


if __name__ == '__main__':
    main()
