# EyeBench AI Coding Agent Instructions

## Overview
EyeBench is a comprehensive benchmark for predicting cognitive states from eye movement data using machine learning. It supports multiple datasets (OneStop, CopCo, SBSAT, PoTeC, MECOL2, IITBHGC), tasks (reading comprehension, dyslexia detection, vocabulary knowledge), and models (transformer-based RoBERTeye, classical ML, deep learning architectures).

## Core Architecture

### Configuration System (Hydra-based)
- **Pattern**: All models/datasets/trainers are dataclasses with `@register_*` decorators
- **Key files**: `src/configs/models/dl/*.py`, `src/configs/data.py`, `src/configs/trainers.py`
- **Factory pattern**: `ModelFactory.get()`, `DataModuleFactory.get()` for instantiation
- **Example**: Adding new model requires: dataclass in `src/configs/models/dl/`, register with `@register_model_config`, add to `ModelNames` enum

### Data Flow Architecture
```
Raw Data → Preprocessing Pipeline → Cross-validation Folds → DataModules → Models
```

1. **Preprocessing**: `src/data/preprocessing/get_data.sh` orchestrates: download → union → preprocess → create_folds → stats
2. **Data structure**: Each dataset has `{Raw,Processed,folds_metadata,folds}` with train/val/test splits per regime
3. **DataModules**: Inherit from `ETDataModule`, implement `create_etdataset()` method
4. **Datasets**: Inherit from `ETDataset`, handle eye features + text tokenization

### Model Architecture Patterns

#### Deep Learning Models (src/models/)
- **Base**: All inherit from `BaseModel` (PyTorch Lightning), implement `forward()` and `shared_step()`
- **RoBERTeye**: Primary model family with word-level vs fixation-level variants (`RoberteyeWord`/`RoberteyeFixation`)
- **Multimodal**: Eye data projected via MAG module or simple linear layer, concatenated with text embeddings
- **Features**: Eye movement features (dwell time, fixation count) + linguistic features (surprisal, frequency)

#### Classical ML Models (src/models/models_ml.py)
- **Pattern**: Inherit from `BaseMLModel`, feature engineering in `configure_features()`
- **Features**: Aggregated eye movement statistics (mean, std, median, etc.) across trials

### Training & Evaluation Workflows

#### Single Run
```bash
python src/run/single_run/train.py +trainer=TrainerDL +model=RoberteyeWord +data=OneStop_TRC
```

#### Multi-Run Pipeline (Production)
1. **Model checker**: `src/run/multi_run/model_checker.sh` - runs train+test for all model/data combinations
2. **Sweep generation**: `src/run/multi_run/sweep_wrapper.sh` - creates hyperparameter sweeps
3. **Results processing**: `raw_to_processed_results.py` → `csv_to_latex.py` for paper tables

#### Cross-validation Structure
- **Folds**: Each dataset has multiple folds with train/val/test splits
- **Regimes**: Different data splits (unseen_subject_seen_item, seen_subject_unseen_item, etc.)
- **Logging**: Results saved to `results/raw/{model_path}/fold_index_{i}/trial_level_test_results.csv`

## Development Patterns

### Adding New Models
1. Create config class in `src/configs/models/dl/NewModel.py` with `@register_model_config`
2. Add to `DLModelNames` enum in `src/configs/constants.py`
3. Implement model class in `src/models/new_model.py` with `@register_model` decorator
4. Add to `supported_models` list in `src/run/multi_run/model_task_matrix.py`

### Adding New Datasets
1. Create dataset class inheriting from `ETDataset` in `src/data/datasets/`
2. Create datamodule inheriting from `ETDataModule` in `src/data/datamodules/`
3. Add data args class in `src/configs/data.py`
4. Update preprocessing pipeline and constants

### Feature Engineering
- **Eye features**: Defined per model in config (e.g., `eye_features`, `fixation_features`)
- **Linguistic features**: Word-level features like surprisal, frequency, POS tags
- **Scaling**: RobustScaler/StandardScaler applied per dataset split to prevent leakage

## Key Commands & Workflows

### Data Preparation
```bash
bash src/data/preprocessing/get_data.sh  # All datasets
bash src/data/preprocessing/get_data.sh DATASET1,DATASET2  # Specific datasets
```

### Model Training/Testing Pipeline
```bash
# Quick model validation (recommended for development)
bash src/run/multi_run/model_checker.sh --data_tasks CopCo_TYP --folds 0,1 --cuda 0

# Full evaluation pipeline
bash src/run/multi_run/test_wrapper_creator.sh  # Generates test scripts
bash test_dl_wrapper.sh  # Runs generated tests
```

### Results Processing
```bash
python src/run/multi_run/raw_to_processed_results.py  # Aggregate results
python src/run/multi_run/csv_to_latex.py  # Generate paper tables
```

## Critical Development Notes

- **CUDA handling**: Always set `CUDA_VISIBLE_DEVICES` before training
- **WandB integration**: Required for tracking, use `wandb.init()` in training scripts
- **Memory considerations**: Eye movement sequences can be very long, use appropriate batching
- **Cross-validation**: Never mix data between folds - strict subject/item separation
- **Feature leakage**: Scalers fitted only on training data, applied to val/test

## Project Structure Conventions
- `src/configs/`: All configuration dataclasses and enums
- `src/data/`: DataModules, Datasets, preprocessing scripts
- `src/models/`: Model implementations (DL and ML)
- `src/run/`: Training/testing scripts (single_run vs multi_run)
- `results/`: Raw trial-level results and processed benchmark tables
- `logs/`: Completed/failed run tracking for multi-run experiments

When implementing features, follow the existing factory pattern and registration decorators. The codebase emphasizes reproducibility and cross-validation rigor for eye movement research.