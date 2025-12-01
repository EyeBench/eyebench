"""Generate interactive HTML tables for benchmark metric CSV files."""

from __future__ import annotations

import csv
import html
import re
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'results' / 'formatted_eyebench_benchmark_results'
OUTPUT_PATH = Path('results/eyebench_metric_tables.md')
PAGE_TITLE = 'EyeBench Metric Dashboards'


FILE_PRIORITY: dict[str, int] = {
    'aggregated_results_test_all_metrics': 0,
    'aggregated_results_val_all_metrics': 1,
    'aggregated_results_test_rankings': 2,
    'aggregated_results_val_rankings': 3,
    'results_tasks_combined': 4,
}

TITLE_OVERRIDES: dict[str, str] = {
    'aggregated_results_test_all_metrics': 'Overall Leaderboard (Test)',
}

BLURB_OVERRIDES: dict[str, str] = {
    'aggregated_results_test_all_metrics': (
        'Macro-level comparison across every benchmark task on the held-out test folds.'
    ),
    'aggregated_results_val_all_metrics': (
        'Cross-validation summary combining all validation folds before final testing.'
    ),
    'results_tasks_combined': (
        'Side-by-side view of the primary metric for each EyeBench task, averaged over folds.'
    ),
}

TAB_LABELS: dict[str, str] = {
    '': 'All',
    'all': 'All',
    'test': 'Test',
    'val': 'Validation',
    'validation': 'Validation',
    'dev': 'Dev',
    'train': 'Train',
}


def sanitize(value: str | None) -> str:
    """Prepare a CSV cell for safe HTML display."""

    if value is None:
        return ''

    cleaned = str(value)
    cleaned = cleaned.replace('\\pm', '±')
    cleaned = cleaned.replace('~', ' ')
    cleaned = cleaned.replace('\\checkmark', '✓')
    cleaned = re.sub(r'\\citep?\{([^}]+)\}', r'[\1]', cleaned)
    cleaned = re.sub(r'\\textbf\{([^}]+)\}', r'\1', cleaned)
    cleaned = re.sub(r'\\textit\{([^}]+)\}', r'\1', cleaned)
    cleaned = cleaned.strip()
    return html.escape(cleaned, quote=False)


def normalise_header(header: str) -> str:
    """Format CSV headers into human-friendly column names."""

    header_clean = header.strip()
    if not header_clean:
        return header_clean

    lower = header_clean.lower()
    if lower == 'data':
        return 'Task'
    if lower == 'model':
        return 'Model'
    return header_clean.replace('_', ' ').title()


def render_table_html(
    table_id: str, column_headers: Iterable[str], rows_data: Iterable[Iterable[str]]
) -> str:
    """Render a DataTables-compatible HTML fragment."""

    header_html = ''.join(f'<th>{col}</th>' for col in column_headers)
    body_rows = []
    for row_values in rows_data:
        cells = ''.join(f'<td>{value}</td>' for value in row_values)
        body_rows.append(f'<tr>{cells}</tr>')
    body_html = ''.join(body_rows)
    return (
        f"<table id='{table_id}' class='display compact stripe eyebench-datatable' "
        "data-datatable='true'>"
        f'<thead><tr>{header_html}</tr></thead>'
        f'<tbody>{body_html}</tbody>'
        '</table>'
    )


def unique_table_id(stem: str, label: str, used_ids: set[str]) -> str:
    """Generate a stable HTML id per table."""

    base = re.sub(r'[^a-z0-9_-]', '-', f'{stem}-{label}'.lower()).strip('-')
    if not base:
        base = 'table'

    candidate = base
    suffix = 1
    while candidate in used_ids:
        suffix += 1
        candidate = f'{base}-{suffix}'

    used_ids.add(candidate)
    return candidate


def add_assets(doc) -> None:
    """Write the static CSS/JS includes required for DataTables."""

    doc.write(
        """<!-- DataTables assets injected by gen_metric_tables.py -->
<link rel='stylesheet' href='https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css'>
<style>
    /* Custom styling for DataTables in Material theme */
    .eyebench-datatable {
        width: 100% !important;
        margin: 1em 0 !important;
        border-collapse: collapse !important;
        font-size: 0.9em !important;
    }
    .eyebench-datatable thead {
        background-color: var(--md-primary-fg-color, white) !important;
        color: black !important;
    }
    .eyebench-datatable thead th {
        padding: 12px 8px !important;
        text-align: left !important;
        font-weight: 600 !important;
        border-bottom: 2px solid var(--md-primary-fg-color, #3f51b5) !important;
    }
    .eyebench-datatable tbody tr {
        border-bottom: 1px solid #ddd !important;
    }
    .eyebench-datatable tbody tr:hover {
        background-color: #f5f5f5 !important;
    }
    .eyebench-datatable tbody td {
        padding: 10px 8px !important;
    }
    .eyebench-datatable tbody tr:nth-child(even) {
        background-color: #fafafa !important;
    }
    /* DataTables filter/search styling */
    .dataTables_wrapper .dataTables_filter {
        float: right !important;
        text-align: right !important;
        margin-bottom: 1em !important;
    }
    .dataTables_wrapper .dataTables_filter input {
        margin-left: 0.5em !important;
        padding: 5px 10px !important;
        border: 1px solid #ddd !important;
        border-radius: 4px !important;
    }
    /* Override Material theme table styles */
    .md-typeset table:not([class]) {
        display: table !important;
    }
    .dataTables_wrapper {
        margin: 2em 0 !important;
    }
</style>
<script src='https://code.jquery.com/jquery-3.7.0.min.js'></script>
<script src='https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js'></script>
<script>
(function(){
    function initialiseTables(){
        if (!window.jQuery || !window.jQuery.fn || !window.jQuery.fn.DataTable){
            setTimeout(initialiseTables, 100);
            return;
        }
        document.querySelectorAll('table[data-datatable="true"]').forEach(function(tbl){
            if (!window.jQuery.fn.dataTable.isDataTable(tbl)){
                window.jQuery(tbl).DataTable({
                    paging: true,
                    pageLength: 25,
                    lengthChange: false,
                    info: true,
                    searching: true,
                    order: [],
                    autoWidth: false,
                    language: {
                        search: "Filter models:",
                        info: "Showing _START_ to _END_ of _TOTAL_ models",
                        infoEmpty: "No models found",
                        infoFiltered: "(filtered from _MAX_ total)",
                        paginate: {
                            first: "First",
                            last: "Last",
                            next: "Next",
                            previous: "Previous"
                        }
                    }
                });
            }
        });
    }
    if (document.readyState === 'loading'){
        document.addEventListener('DOMContentLoaded', function(){
            setTimeout(initialiseTables, 100);
        });
    } else {
        setTimeout(initialiseTables, 100);
    }
})();
</script>
"""
    )


def bucket_rows(
    reader: csv.DictReader, eval_column: str | None
) -> OrderedDict[str, list[dict[str, str]]]:
    """Group CSV rows by evaluation split if provided."""

    buckets: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for record in reader:
        key = ''
        if eval_column:
            key = (record.get(eval_column, '') or '').strip().lower()
        buckets.setdefault(key, []).append(record)
    return buckets


def sort_metric_files(files: Iterable[Path]) -> list[Path]:
    """Sort CSV files so headline tables appear first."""

    def sort_key(path: Path) -> tuple[int, str]:
        stem = path.stem.lower()
        return (FILE_PRIORITY.get(stem, 100), stem)

    return sorted(files, key=sort_key)


def write_placeholder(message: str) -> None:
    """Emit a lightweight placeholder page when metric exports are missing."""

    with mkdocs_gen_files.open(OUTPUT_PATH, 'w') as doc:
        doc.write(f'# {PAGE_TITLE}\n\n')
        doc.write(message.rstrip() + '\n')

    mkdocs_gen_files.set_edit_path(OUTPUT_PATH, Path('docs/gen_metric_tables.py'))


def main() -> None:
    if not DATA_DIR.exists():
        write_placeholder(
            'Metric exports are unavailable. Run the EyeBench evaluation pipeline to populate '
            '`results/formatted_eyebench_benchmark_results/` before building the docs.'
        )
        return

    metric_files = [
        path for path in DATA_DIR.glob('*.csv') if not path.name.startswith('stats_')
    ]

    if not metric_files:
        write_placeholder(
            'No formatted benchmark CSV files were found. Run the evaluation pipeline before '
            'building the documentation.'
        )
        return

    used_ids: set[str] = set()

    with mkdocs_gen_files.open(OUTPUT_PATH, 'w') as doc:
        doc.write(f'# {PAGE_TITLE}\n\n')
        doc.write(
            'Interactive tables sourced from the latest formatted benchmark exports. Values show '
            'the mean and standard deviation across folds.\n\n'
        )

        add_assets(doc)

        for csv_path in sort_metric_files(metric_files):
            stem = csv_path.stem.lower()
            title = TITLE_OVERRIDES.get(stem, stem.replace('_', ' ').title())
            blurb = BLURB_OVERRIDES.get(stem)

            doc.write(f'## {title}\n\n')
            if blurb:
                doc.write(f'*{blurb}*\n\n')
            else:
                doc.write(f'*Source file: `{csv_path.name}`*\n\n')

            with csv_path.open('r', encoding='utf-8') as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames or []
                eval_column = None
                for candidate in ('Eval Type', 'Split', 'Subset'):
                    if candidate in fieldnames:
                        eval_column = candidate
                        break

                buckets = bucket_rows(reader, eval_column)

            for eval_key, bucket_rows_list in buckets.items():
                if eval_column:
                    label = TAB_LABELS.get(
                        eval_key, eval_key.title() if eval_key else 'All'
                    )
                    doc.write(f'=== "{label}"\n\n')

                column_keys = [h for h in fieldnames if h != eval_column]
                display_headers = [normalise_header(key) for key in column_keys]

                rows_for_table = [
                    [sanitize(record.get(key, '')) for key in column_keys]
                    for record in bucket_rows_list
                ]

                table_id = unique_table_id(stem, eval_key or 'all', used_ids)
                table_html = render_table_html(
                    table_id, display_headers, rows_for_table
                )
                doc.write(table_html)
                doc.write('\n\n')

            doc.write('\n')

    mkdocs_gen_files.set_edit_path(OUTPUT_PATH, Path('docs/gen_metric_tables.py'))


# Execute main function when imported by MkDocs gen-files plugin
main()
