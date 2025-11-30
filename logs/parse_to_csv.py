import os
import re

import pandas as pd

failed_log_file = 'logs/failed_runs.log'
completed_log_file = 'logs/completed_runs.log'

failed_csv = 'logs/failed_runs.csv'
all_csv = 'logs/all_runs.csv'
output_dir = 'logs/by_data_task'
os.makedirs(output_dir, exist_ok=True)

# Regex patterns (same for both)
pattern1 = re.compile(
    r'(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2}) '
    r'Data Task: (?P<data_task>.*?), Trainer: (?P<trainer>.*?), Fold: (?P<fold>\d+), '
    r'stage: (?P<stage>\w+), Model: (?P<model>.*)'
)

pattern2 = re.compile(
    r'(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2}) '
    r'Model: (?P<model>.*?), Data Task: (?P<data_task>.*?), Trainer: (?P<trainer>.*?), '
    r'Fold: (?P<fold>\d+), stage: (?P<stage>\w+)'
)


def parse_log_file(file_path, success_flag):
    data = []
    if not os.path.exists(file_path):
        print(f'⚠️ File not found: {file_path}')
        return data

    with open(file_path, 'r') as f:
        for line in f:
            match = pattern1.search(line) or pattern2.search(line)
            if match:
                entry = match.groupdict()
                entry['success'] = success_flag
                data.append(entry)
    return data


# Parse logs
failed_data = parse_log_file(failed_log_file, '!!! Failed !!!')
completed_data = parse_log_file(completed_log_file, 'V')

# Create DataFrames
df_failed = pd.DataFrame(failed_data)
df_completed = pd.DataFrame(completed_data)

# Ensure correct data types
for df in [df_failed, df_completed]:
    if not df.empty:
        df['fold'] = df['fold'].astype(int)

sort_by = ['data_task', 'stage', 'trainer', 'model', 'fold']

# ✅ Save failed CSV
if not df_failed.empty:
    df_failed = df_failed.sort_values(by=sort_by)
    df_failed.to_csv(failed_csv, index=False)
    print(f'✅ Saved failed CSV to {failed_csv}')

# ✅ Combine failed + completed and save
df_all = pd.concat([df_failed, df_completed], ignore_index=True)
if not df_all.empty:
    df_all = df_all.sort_values(by=sort_by)
    df_all.to_csv(all_csv, index=False)
    print(f'✅ Saved combined CSV to {all_csv}')


# ✅ Save separate CSVs by data_task
def save_by_data_task(df, prefix):
    for task, df_task in df.groupby('data_task'):
        task_name = task.replace(' ', '_').replace('/', '_')
        output_path = os.path.join(output_dir, f'{prefix}_{task_name}.csv')
        df_task.to_csv(output_path, index=False)
        print(f"✅ Saved {prefix} for data_task '{task}' → {output_path}")


if not df_failed.empty:
    save_by_data_task(df_failed, 'failed')

if not df_all.empty:
    save_by_data_task(df_all, 'all')
