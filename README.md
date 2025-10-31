# Manufacturing Parameter Optimization System

MLOps-compatible system for predicting and optimizing manufacturing parameters using deep learning.

## Project Structure

```
kemal_app/
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architecture and training
│   ├── inference/         # Model inference and optimization
│   ├── config/            # Configuration management
│   ├── utils/             # Utility functions
│   └── gui/               # GUI module (to be enhanced)
├── models/                # Saved models directory
├── data/                  # Data directory
├── train.py              # Training entry point
├── predict.py            # Prediction entry point
├── requirements.txt      # Python dependencies
├── setup.py              # Package setup
└── pyproject.toml        # Modern Python project configuration
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install as a package:
```bash
pip install -e .
```

## Usage

### CLI Tool (Recommended)

A comprehensive CLI tool is available with multiple commands for different operations:

#### List Available Conditions
```bash
python cli.py list-conditions
```

#### Train a Model
```bash
python cli.py train
python cli.py train --verbose  # For detailed training output
```

#### Make Predictions

Single prediction:
```bash
python cli.py predict --Vc 125.0 --fn 0.125 --condition Dry
python cli.py predict --Vc 125.0 --fn 0.125 --condition Dry --output result.json
```

Batch predictions from CSV file (must contain columns: Vc, fn, Condition):
```bash
python cli.py predict-batch --input data.csv --output predictions.csv
```

#### Optimize Parameters

Optimize for a single condition:
```bash
python cli.py optimize --condition Dry
python cli.py optimize --condition Dry --output optimal_params.json
```

Optimize for all conditions:
```bash
python cli.py optimize --all
```

With custom bounds:
```bash
python cli.py optimize --condition Dry --Vc-bounds 100 150 --fn-bounds 0.10 0.15
```

#### Compare Conditions

Compare predictions across different conditions:
```bash
python cli.py compare --Vc 125.0 --fn 0.125 --all
python cli.py compare --Vc 125.0 --fn 0.125 --conditions Dry MQL Hybrid
python cli.py compare --Vc 125.0 --fn 0.125 --condition Dry
```

#### Show Model Information
```bash
python cli.py info
```

#### Get Help
```bash
python cli.py --help
python cli.py <command> --help  # Help for specific command
```

### Alternative Entry Points

#### Training

Train a new model using the original script:
```bash
python train.py
```

This will:
- Load and preprocess the data
- Train the MLP model
- Save the model, preprocessor, and metadata to the `models/` directory

#### Prediction

Make a single prediction:
```bash
python predict.py --Vc 125.0 --fn 0.125 --condition Dry
```

Optimize parameters:
```bash
python predict.py --condition Dry --optimize
```

### Command Line Options (predict.py)

```bash
python predict.py --help
```

Options:
- `--model-path`: Path to model file (default: uses config)
- `--preprocessor-path`: Path to preprocessor file
- `--Vc`: Cutting speed (m/min)
- `--fn`: Feed rate (mm/rev)
- `--condition`: Manufacturing condition (Dry, MQL, Hybrid, Cryo, NF-1, NF-2)
- `--optimize`: Optimize parameters instead of predicting

### Available Manufacturing Conditions

- `Dry`: Kuru İşleme (Dry Processing)
- `MQL`: MQL (Minimum Yağlama - Minimum Quantity Lubrication)
- `Hybrid`: Hibrit Yağlama (Hybrid Lubrication)
- `Cryo`: Kriyojenik Soğutma (Cryogenic Cooling)
- `NF-1`: Nanofluid 1
- `NF-2`: Nanofluid 2

## API Usage

The system can be integrated as a Python package:

```python
from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.models.trainer import ModelTrainer
from src.inference.predictor import Predictor
from src.inference.optimizer import ParameterOptimizer
from src.config.config import get_config

# Load configuration
config = get_config()

# Train model
data_loader = DataLoader()
X = data_loader.get_features()
y = data_loader.get_targets()

preprocessor = Preprocessor()
X_processed = preprocessor.fit_transform(X)

trainer = ModelTrainer(preprocessor=preprocessor)
trainer.train(X_processed, y.values, input_dim=X_processed.shape[1])
trainer.save_model()

# Use for inference
predictor = Predictor(
    model_path="models/mlp_model.keras",
    preprocessor_path="models/mlp_model_preprocessor.joblib"
)

T, E = predictor.predict(Vc=125.0, fn=0.125, condition='Dry')

# Optimize
optimizer = ParameterOptimizer(predictor)
result = optimizer.optimize(condition='Dry')
```

## MLOps Compatibility

This structure is designed to be compatible with MLOps pipelines:

- **Modular architecture**: Separated concerns (data, models, inference)
- **Configuration management**: Centralized config system
- **Reproducibility**: Saved models, preprocessors, and metadata
- **CLI interfaces**: Command-line entry points for automation
- **Package structure**: Installable as a Python package
- **Version control**: Proper `.gitignore` for models and data

## Model Architecture

- **Type**: Multi-Layer Perceptron (MLP)
- **Input**: Preprocessed features (Vc, fn, Condition)
- **Output**: 2 values (T: Tool Life, E: Energy)
- **Architecture**: 64 → 32 → 2 neurons
- **Activation**: ReLU for hidden layers, Linear for output
- **Loss**: Mean Squared Error (MSE)

## Configuration

Default configuration can be found in `src/config/config.py`. To use a custom configuration file:

```python
config = get_config("path/to/config.json")
```

## Development

The project follows software engineering best practices:

- Clean separation of concerns
- Type hints where appropriate
- Error handling
- Documentation strings
- Modular, testable code structure

## Future Enhancements

- Enhanced GUI module
- API server (Flask/FastAPI)
- Model versioning
- Experiment tracking (MLflow, Weights & Biases)
- Unit tests
- CI/CD pipeline integration

