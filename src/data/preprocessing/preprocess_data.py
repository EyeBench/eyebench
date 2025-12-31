"""
Preprocess data for different datasets.
This script standardizes column names, filters data, and calculates text metrics.
It is designed to be modular, allowing for easy addition of new datasets and processing steps.
"""

import argparse

from loguru import logger
from tqdm import tqdm

from src.configs.constants import DataSets
from src.configs.data import DataArgs, get_data_args
from src.data.preprocessing.dataset_preprocessing.base import DatasetProcessor

tqdm.pandas()

logger.add('logs/preprocessing.log', level='INFO')

from src.data.preprocessing.dataset_preprocessing.copco import (  # noqa: E402
    CopCoProcessor,
)
from src.data.preprocessing.dataset_preprocessing.iitbhgc import (  # noqa: E402
    IITBHGCProcessor,
)
from src.data.preprocessing.dataset_preprocessing.meco import (  # noqa: E402
    MECOProcessor,
)
from src.data.preprocessing.dataset_preprocessing.onestop import (  # noqa: E402
    OneStopProcessor,
)
from src.data.preprocessing.dataset_preprocessing.potec import (  # noqa: E402
    PoTeCProcessor,
)
from src.data.preprocessing.dataset_preprocessing.sbsat import (  # noqa: E402
    SBSATProcessor,
)

PROCESSOR_REGISTRY = {
    DataSets.HALLUCINATION: IITBHGCProcessor,
    DataSets.ONESTOP: OneStopProcessor,
    DataSets.COPCO: CopCoProcessor,
    DataSets.POTEC: PoTeCProcessor,
    DataSets.SBSAT: SBSATProcessor,
    DataSets.MECO_L2W1: MECOProcessor,
    DataSets.MECO_L2W2: MECOProcessor,
    DataSets.MECO_L2: MECOProcessor,
}


def get_processor(data_args: DataArgs) -> DatasetProcessor:
    processor_class = PROCESSOR_REGISTRY.get(data_args.dataset_name)

    if processor_class:
        return processor_class(data_args)

    logger.info(
        f'Unknown dataset: {data_args.dataset_name}, using default processor',
    )
    return DatasetProcessor(data_args)


def main() -> int:
    """
    Main function to execute dataset processing.
    Configures the data settings and initiates the data processing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='')
    args = parser.parse_args()
    dataset = args.dataset

    datasets = dataset.split(',') if dataset else list(DataSets)

    for dataset_name in datasets:
        logger.info(f'Processing {dataset_name}...')
        data_args = get_data_args(dataset_name)
        if not data_args:
            logger.warning(
                f'No data args found for {dataset_name}. Skipping...',
            )
            continue
        try:
            processor = get_processor(data_args)
            processed_data = processor.process()
            processor.save_processed_data(processed_data=processed_data)
            logger.info(f'Finished processing {dataset_name}')
        except FileNotFoundError as e:
            logger.warning(f'FileNotFoundError processing {dataset_name}: {e}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
