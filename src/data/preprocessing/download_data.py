import argparse
from pathlib import Path

import pymovements as pm
from loguru import logger
from tqdm import tqdm

from src.configs.constants import DataSets

logger.add('logs/preprocessing.log', level='INFO')


def main() -> int:
    data_path = Path('data')
    data_path.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='')
    args = parser.parse_args()

    dataset = args.dataset

    if dataset:
        datasets_names = dataset.split(',')
    else:
        datasets_names = [
            DataSets.ONESTOP,
            DataSets.COPCO,
            DataSets.POTEC,
            DataSets.SBSAT,
            DataSets.HALLUCINATION,
            # DataSets.MECO_L2W1,
            # DataSets.MECO_L2W2,
            DataSets.MECO_L2,
        ]

    for dataset_name in tqdm(
        datasets_names,
        desc='Downloading datasets',
        unit='dataset',
        total=len(datasets_names),
    ):
        try:
            if dataset_name == DataSets.MECO_L2:
                # exclude 'gaze' and 'reading measure' files from download
                dataset_def_w1 = pm.DatasetLibrary.get(f'{dataset_name}W1')
                dataset_def_w1.has_files['gaze'] = False
                dataset_def_w2 = pm.DatasetLibrary.get(f'{dataset_name}W2')
                dataset_def_w2.has_files['gaze'] = False
                pm.Dataset(dataset_def_w1, data_path / DataSets.MECO_L2W1).load()
                pm.Dataset(dataset_def_w2, data_path / DataSets.MECO_L2W2).load()
            else:
                dataset_def = pm.DatasetLibrary.get(dataset_name)
                dataset_def.has_files['gaze'] = False
                pm.Dataset(dataset_def, data_path / dataset_name).load()
            logger.info(f'{dataset_name} already downloaded. Continuing...')
        except Exception:
            logger.info(f'{dataset_name} not downloaded yet. Downloading...')
            if dataset_name == DataSets.MECO_L2:
                dataset_def_w1 = pm.DatasetLibrary.get(f'{dataset_name}W1')
                dataset_def_w1.has_files['gaze'] = False
                dataset_def_w2 = pm.DatasetLibrary.get(f'{dataset_name}W2')
                dataset_def_w2.has_files['gaze'] = False
                pm.Dataset(dataset_def_w1, data_path / DataSets.MECO_L2W1).download()
                pm.Dataset(dataset_def_w2, data_path / DataSets.MECO_L2W2).download()
            else:
                dataset_def = pm.DatasetLibrary.get(dataset_name)
                dataset_def.has_files['gaze'] = False
                pm.Dataset(dataset_def, data_path / dataset_name).download()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
