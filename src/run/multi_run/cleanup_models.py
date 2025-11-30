import argparse
import re
from pathlib import Path


def get_non_lowest_checkpoint_paths(
    search_path: Path, checkpoint_template: str, keep_one_lowest: bool = False
) -> list[Path]:
    """
    Find checkpoint files and return all except those with the lowest score.

    Args:
        search_path: Path to search for checkpoints
        checkpoint_template: Template string to match checkpoint files
        keep_one_lowest: If True, keep only one checkpoint with the lowest score

    Returns:
        List of Path objects for checkpoints that don't have the lowest score
    """
    full_template = f'*{checkpoint_template}*.ckpt'
    checkpoint_files = list(search_path.glob(full_template))

    if not checkpoint_files:
        return []

    # Define regex pattern once to avoid repetition
    pattern = rf'{checkpoint_template}-(\d+\.\d+)(-v\d+)?\.ckpt$'

    def extract_score(file_path: Path) -> float:
        """Extract the score from a checkpoint filename or return infinity if not found."""
        match = re.search(pattern, str(file_path.name))
        if match:
            return float(match.group(1))
        return float('inf')

    checkpoint_files = sorted(checkpoint_files, key=extract_score)

    # Find the minimum score
    min_score = extract_score(checkpoint_files[0])

    # Keep all models with the minimum score
    lowest_checkpoints = [f for f in checkpoint_files if extract_score(f) == min_score]

    if keep_one_lowest and lowest_checkpoints:
        # Keep only one of the lowest models
        lowest_checkpoints = [lowest_checkpoints[0]]

    # Return all files except the ones with the lowest score
    return [f for f in checkpoint_files if f not in lowest_checkpoints]


def process_checkpoints(
    search_path: Path,
    checkpoint_template: str,
    keep_one_lowest: bool = False,
    real_run: bool = False,
) -> list[float]:
    """
    Process checkpoints in the given path and return sizes of non-lowest checkpoints.

    Args:
        search_path: Base path to search for checkpoints
        checkpoint_template: Template string to match checkpoint files
        keep_one_lowest: If True, keep only one checkpoint with the lowest score
        real_run: If True, delete non-lowest checkpoints instead of just reporting

    Returns:
        List of sizes (in GB) of the non-lowest checkpoints
    """
    total_sizes = []
    for subfolder in search_path.glob(pattern='*'):
        if not subfolder.is_dir():
            continue

        for sub_subfolder in subfolder.glob(pattern='fold_index=*'):
            if not sub_subfolder.is_dir():
                continue

            non_lowest_checkpoints = get_non_lowest_checkpoint_paths(
                search_path=sub_subfolder,
                checkpoint_template=checkpoint_template,
                keep_one_lowest=keep_one_lowest,
            )

            for checkpoint in non_lowest_checkpoints:
                size = checkpoint.stat().st_size / (1024 * 1024 * 1024)  # Convert to GB
                total_sizes.append(size)
                if real_run:
                    print(f'Deleting checkpoint: {checkpoint}')
                    checkpoint.unlink()

    return total_sizes


def count_wandb_runs(search_path: Path) -> None:
    """
    Print the number of wandb runs in each subfolder.

    Args:
        search_path: Base path to search for wandb runs
    """
    for subfolder in search_path.glob('*'):
        if not subfolder.is_dir():
            continue

        for sub_subfolder in subfolder.glob('fold_index=*'):
            if not sub_subfolder.is_dir():
                continue

            count = len(list(sub_subfolder.glob('wandb/*run*')))
            print(f'{sub_subfolder}: {count} wandb runs')


def main():
    """Main function to clean up model checkpoints."""
    parser = argparse.ArgumentParser(
        description='Cleanup model checkpoints, keeping only the best ones.'
    )
    parser.add_argument(
        '--real_run',
        action='store_true',
        help='Actually delete files. Without this flag, only reports what would be deleted.',
    )
    parser.add_argument(
        '--keep_one_lowest',
        action='store_true',
        help='Keep only one of the lowest scoring models (instead of all with same lowest score).',
    )
    parser.add_argument(
        '--print_num_wandb_runs_in_folder',
        action='store_true',
        help='Print the number of wandb runs in each folder.',
    )
    args = parser.parse_args()

    search_paths = [
        Path('.') / 'outputs',
    ]

    checkpoint_templates = [
        'lowest_loss_val_all',
    ]

    for search_path in search_paths:
        for checkpoint_template in checkpoint_templates:
            total_sizes = process_checkpoints(
                search_path=search_path,
                checkpoint_template=checkpoint_template,
                keep_one_lowest=args.keep_one_lowest,
                real_run=args.real_run,
            )

            if total_sizes:
                action = 'Deleted' if args.real_run else 'Would delete'
                print(
                    f'{action} non-lowest checkpoints for {checkpoint_template} in {search_path}: '
                    f'{round(sum(total_sizes), 2)} GB (total {len(total_sizes)} files)'
                )

    if args.print_num_wandb_runs_in_folder:
        for search_path in search_paths:
            count_wandb_runs(search_path)


if __name__ == '__main__':
    main()
