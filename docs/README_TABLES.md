# Generating Static Metric Tables for Documentation

This directory contains scripts for generating the EyeBench metric tables displayed on the documentation website.

## Overview

The metric tables are **pre-generated** and committed to the repository at `docs/results/eyebench_metric_tables.md`.

## Files

- `generate_static_tables.py` - Standalone script to regenerate the metric tables
- `gen_metric_tables.py` - Legacy script (kept for reference, uses mkdocs_gen_files)
- `results/eyebench_metric_tables.md` - Pre-generated markdown file with interactive tables (committed to repo)

## When to Regenerate Tables

Run the generation script whenever benchmark results are updated:

```bash
# From the repository root
python docs/generate_static_tables.py
```

Or from the docs directory:

```bash
# From docs/
python generate_static_tables.py
```

## Process

1. The script reads CSV files from `results/formatted_eyebench_benchmark_results/`
2. It generates an HTML table for each CSV file with DataTables.js integration
3. The output is written to `docs/results/eyebench_metric_tables.md`
4. Commit the updated markdown file to the repository
5. GitHub Actions will deploy the pre-generated file (no dynamic generation needed)


## Deployment

The `mkdocs.yml` configuration references the static file directly:

```yaml
nav:
  - Results:
      - Metric Dashboards: results/eyebench_metric_tables.md
```
