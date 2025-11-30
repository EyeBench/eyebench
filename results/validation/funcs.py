from pathlib import Path

import pandas as pd


def highlight_nsample_differences(
    stats: pd.DataFrame, meta_cols=None, minimal: bool = False
):
    # Create eval_type_regime column (for compact grouping)
    stats = stats.copy()
    stats['eval_type_regime'] = stats['eval_type'] + '_' + stats['eval_regime']

    # Pivot to see n_samples per model
    stats_pivot = stats.pivot_table(
        index=['data_task', 'metric_name', 'fold', 'eval_type_regime'],
        columns='model',
        values='n_samples',
        aggfunc='first',
    ).sort_values(by=['metric_name', 'fold'])

    # Reset index to make metadata accessible as columns
    stats_pivot = stats_pivot.reset_index()

    if meta_cols is None:
        meta_cols = ['data_task', 'metric_name', 'fold', 'eval_type_regime']

    model_cols = [c for c in stats_pivot.columns if c not in meta_cols]
    stats_pivot = stats_pivot.copy()

    # Compute majority value (mode) per row
    stats_pivot['majority_val'] = stats_pivot[model_cols].mode(axis=1)[0]

    # Identify differing cells
    diff_mask = stats_pivot[model_cols].ne(stats_pivot['majority_val'], axis=0)
    stats_pivot['n_unique_n_samples'] = stats_pivot[model_cols].nunique(axis=1)

    if minimal:
        # Keep only rows with differences
        rows_with_diff = diff_mask.any(axis=1)
        df_min = stats_pivot.loc[rows_with_diff, meta_cols + model_cols]
        # Keep only model columns that ever differ
        cols_with_diff = diff_mask.any(axis=0)
        cols_to_keep = meta_cols + list(cols_with_diff[cols_with_diff].index)
        df_to_show = df_min[cols_to_keep]
    else:
        df_to_show = stats_pivot

    # Styling function for highlighting
    def highlight_row(row):
        mode_val = row['majority_val']
        return [
            'color: red; font-weight: bold'
            if (col in model_cols and row[col] != mode_val)
            else ''
            for col in df_to_show.columns
        ]

    styled = df_to_show.style.apply(highlight_row, axis=1).format(precision=0)
    return styled


if __name__ == '__main__':
    results_dir = Path.cwd() / 'results'

    data_tasks = [
        'CopCo_TYP',
        'CopCo_RCS',
        'OneStop_RC',
        'SBSAT_RC',
        'SBSAT_STD',
        'PoTeC_RC',
        'PoTeC_DE',
        'IITBHGC_CV',
        'MECOL2_LEX',
    ]

    data_task = 'PoTeC_DE'

    stats = pd.read_csv(
        results_dir / f'eyebench_benchmark_results/stats_{data_task}.csv'
    )

    # # Highlight differences normally (styled view)
    # highlight_nsample_differences(stats)

    # Show only rows and model columns that contain red values
    minimal_df = highlight_nsample_differences(stats, minimal=True)
