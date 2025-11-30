import datetime

from loguru import logger

logger.add('logs/search_spaces.log', rotation='10 MB')


def count_hyperparameter_configs(
    config: dict, log_specific_values: bool = True, n_hours: int = 24
) -> tuple:
    """
    Traverse an arbitrarily nested dict, find all dicts with a 'values' key
    whose value is a list, and:
        1. Print the full key path, the number of values, and the values themselves
        2. Compute the product of all those lengths
        3. Print the time each run should get if runs are evenly distributed over 24 hours
    """

    counts = {}
    values_map = {}

    def recurse(d: dict, path: str = ''):
        for key, val in d.items():
            current_path = f'{path}.{key}' if path else key
            if isinstance(val, dict):
                if 'values' in val and isinstance(val['values'], list):
                    counts[current_path] = len(val['values'])
                    values_map[current_path] = val['values']
                else:
                    recurse(val, current_path)

    recurse(config)

    if log_specific_values:
        # Print per-parameter counts and their values
        for param, n in counts.items():
            if n > 1:
                logger.info(
                    f'{param.replace(".parameters", "")}: {n} possible values → {values_map[param]}'
                )

    # Compute total combinations
    total = 1
    for n in counts.values():
        total *= n

    # Compute time per run over a 24-hour period
    seconds = n_hours * 3600
    time_per_run = seconds / total if total else 0
    # Format as DD:HH:MM:SS
    td = datetime.timedelta(seconds=round(time_per_run))
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted = f'{days:02}:{hours:02}:{minutes:02}:{seconds:02}'
    logger.info(f'Total number of runs: {total}. Time per run: {formatted}')
    return formatted, int(total)


if __name__ == '__main__':
    from src.run.multi_run.search_spaces import search_space_by_model

    logger.info(len(search_space_by_model))
    for model, cfg in search_space_by_model.items():
        logger.info(f'--- {model} ---')
        count_hyperparameter_configs(cfg, log_specific_values=True, n_hours=24)
