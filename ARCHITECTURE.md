# Architecture Documentation

## Overview

This project follows MLOps best practices with a modular architecture that separates concerns and enables easy integration into larger systems.

## Module Structure

### 1. Data Module (`src/data/`)

**Purpose**: Data loading and preprocessing

- **`data_loader.py`**: Handles loading manufacturing data from various sources
  - Internal data storage
  - File-based loading (CSV)
  - Data validation
  - Condition filtering

- **`preprocessor.py`**: Feature engineering and transformation
  - StandardScaler for numerical features
  - OneHotEncoder for categorical features
  - Save/load functionality for production use

### 2. Models Module (`src/models/`)

**Purpose**: Model architecture and training

- **`model_builder.py`**: Defines model architectures
  - MLP model construction
  - Configurable architecture (hidden layers, activation, etc.)
  - Model summary utilities

- **`trainer.py`**: Training orchestration
  - TensorFlow configuration
  - Training pipeline
  - Model persistence
  - Training metadata saving

### 3. Inference Module (`src/inference/`)

**Purpose**: Model prediction and optimization

- **`predictor.py`**: Model loading and prediction
  - Single and batch predictions
  - Model and preprocessor loading
  - Ready-state checking

- **`optimizer.py`**: Parameter optimization
  - Scipy-based optimization
  - Configurable bounds and objectives
  - E/T ratio minimization

### 4. Config Module (`src/config/`)

**Purpose**: Configuration management

- **`config.py`**: Centralized configuration
  - Default values
  - JSON file support
  - Dictionary-like interface

### 5. Utils Module (`src/utils/`)

**Purpose**: Utility functions

- **`system_config.py`**: System-level configuration
  - Thread count optimization
  - Library environment variables

### 6. GUI Module (`src/gui/`)

**Purpose**: User interface (placeholder for future enhancement)

Currently a stub module to be implemented later.

## Entry Points

### `train.py`
- Training entry point
- Loads data → preprocesses → trains → saves
- CLI-compatible for automation

### `predict.py`
- Inference entry point
- Loads model → makes predictions
- Supports single predictions and optimization
- Command-line interface

### `example_usage.py`
- Demonstrates API usage
- Shows training, prediction, and optimization workflows
- Useful for integration testing

## Data Flow

```
┌─────────────┐
│ DataLoader  │ → DataFrame
└─────────────┘
      │
      ▼
┌─────────────┐
│Preprocessor │ → NumPy Array
└─────────────┘
      │
      ▼
┌─────────────┐
│ ModelTrainer│ → Trained Model
└─────────────┘
      │
      ▼
┌─────────────┐
│  Predictor  │ → Predictions
└─────────────┘
      │
      ▼
┌─────────────┐
│ Optimizer   │ → Optimal Parameters
└─────────────┘
```

## Integration Points

### For MLOps Pipelines

1. **Training Pipeline**:
   ```bash
   python train.py
   ```
   - Output: Saved model, preprocessor, metadata

2. **Inference Service**:
   ```python
   from src.inference.predictor import Predictor
   predictor = Predictor(model_path, preprocessor_path)
   T, E = predictor.predict(Vc, fn, condition)
   ```

3. **Configuration**:
   ```python
   from src.config.config import get_config
   config = get_config("config.json")
   ```

### For Software Integration

- **Import as package**: `from src.models.trainer import ModelTrainer`
- **CLI interface**: `python train.py`, `python predict.py`
- **Configurable**: JSON configuration files
- **Extensible**: Modular structure allows easy extension

## File Structure

```
kemal_app/
├── src/                    # Source code (installable package)
│   ├── __init__.py
│   ├── data/              # Data handling
│   ├── models/            # Model training
│   ├── inference/         # Model inference
│   ├── config/            # Configuration
│   ├── utils/             # Utilities
│   └── gui/               # GUI (future)
├── models/                # Saved models (gitignored)
├── data/                  # Data files (gitignored)
├── train.py              # Training entry point
├── predict.py            # Prediction entry point
├── example_usage.py      # API examples
├── requirements.txt      # Dependencies
├── setup.py              # Package setup
├── pyproject.toml        # Modern Python config
└── README.md             # User documentation
```

## Best Practices Implemented

1. **Separation of Concerns**: Each module has a single responsibility
2. **Dependency Injection**: Configurable components
3. **Error Handling**: Proper exception handling throughout
4. **Type Hints**: Type annotations for better IDE support
5. **Documentation**: Docstrings for all public methods
6. **Configuration Management**: Centralized config system
7. **Model Versioning**: Saved models with metadata
8. **Reproducibility**: Saved preprocessors ensure consistent transformations

## Testing Strategy (Future)

- Unit tests for each module
- Integration tests for full pipeline
- Model validation tests
- Performance benchmarks

## Deployment Considerations

- Models are saved separately (not in git)
- Preprocessors saved alongside models
- Metadata for tracking model versions
- CLI interfaces for automation
- Package installation support (`pip install -e .`)

