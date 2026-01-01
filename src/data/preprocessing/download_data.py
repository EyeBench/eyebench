import argparse
from pathlib import Path

import pymovements as pm
import rdata
import requests
from loguru import logger
from tqdm import tqdm

from src.configs.constants import DataSets

logger.add('logs/preprocessing.log', level='INFO')

BASE_OSF_URL = 'https://osf.io/download/'
AUXILIARY_FILES: dict[str, dict[str, str]] = {
    DataSets.MECO_L2: {  # Hosted on MECO L2: The Multilingual Eye-movement COrpus, L2 (English) - https://osf.io/q9h43
        'MECOL2W1/demographics/joint.ind.diff.l2.rda': '4zu8d',
        'MECOL2W2/demographics/joint.ind.diff.l2.w2.rda': 'keuvm',
        'MECOL2/stimuli/texts.meco.l2.rda': 'zwfdb',
    },
}


def download_auxiliary_files(root: Path, dataset_name: str) -> None:
    """Download auxiliary resources not covered by DatasetLibrary for a specific dataset."""
    if dataset_name not in AUXILIARY_FILES:
        return

    for relative_path, resource_id in AUXILIARY_FILES[dataset_name].items():
        destination = root / relative_path
        if destination.exists():
            logger.info(
                f'{relative_path} already present at {destination}. Continuing...'
            )
            continue

        url = f'{BASE_OSF_URL}{resource_id}'
        logger.info(f'Downloading {relative_path} from {url}')
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, 'wb') as fp:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    fp.write(chunk)


def convert_rda_to_csv(root: Path, dataset_name: str) -> None:
    """Convert RDA files to CSV for specific datasets."""
    if dataset_name != DataSets.MECO_L2:
        return
    rda_path = root / 'MECOL2/stimuli/texts.meco.l2.rda'
    csv_path = root / 'MECOL2/stimuli/stimuli.csv'

    if csv_path.exists():
        logger.info(f'{csv_path} already exists. Skipping conversion...')
        return

    if not rda_path.exists():
        logger.warning(f'{rda_path} not found. Skipping conversion...')
        return

    logger.info(f'Converting {rda_path} to {csv_path}')
    rda_data = rdata.read_rda(rda_path)
    df = rda_data['d']
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f'Saved stimuli CSV to {csv_path}')


def prepare_dataset_definition(dataset_name: str):
    """Prepare dataset definition with gaze files disabled."""
    dataset_def = pm.DatasetLibrary.get(dataset_name)
    dataset_def.has_files['gaze'] = False
    return dataset_def


def load_or_download_dataset(
    dataset_name: str, data_path: Path, download: bool = False
) -> None:
    """Load or download a dataset based on the flag."""
    if dataset_name == DataSets.MECO_L2:
        dataset_def_w1 = prepare_dataset_definition(f'{dataset_name}W1')
        dataset_def_w2 = prepare_dataset_definition(f'{dataset_name}W2')
        dataset_w1 = pm.Dataset(dataset_def_w1, data_path / DataSets.MECO_L2W1)
        dataset_w2 = pm.Dataset(dataset_def_w2, data_path / DataSets.MECO_L2W2)
        if download:
            dataset_w1.download()
            dataset_w2.download()
        else:
            dataset_w1.load()
            dataset_w2.load()
    else:
        dataset_def = prepare_dataset_definition(dataset_name)
        dataset = pm.Dataset(dataset_def, data_path / dataset_name)
        if download:
            dataset.download()
        else:
            dataset.load()


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
            DataSets.MECO_L2,
        ]

    for dataset_name in tqdm(
        datasets_names,
        desc='Downloading datasets',
        unit='dataset',
        total=len(datasets_names),
    ):
        try:
            load_or_download_dataset(dataset_name, data_path, download=False)
            logger.info(f'{dataset_name} already downloaded. Continuing...')
        except Exception:
            logger.info(f'{dataset_name} not downloaded yet. Downloading...')
            load_or_download_dataset(dataset_name, data_path, download=True)

        download_auxiliary_files(data_path, dataset_name)
        convert_rda_to_csv(data_path, dataset_name)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
