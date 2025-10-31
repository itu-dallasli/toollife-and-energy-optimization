"""
Model training entry point.

This script trains the MLP model and saves it for later use.
Compatible with MLOps pipelines.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.system_config import configure_system
from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.models.trainer import ModelTrainer
from src.config.config import get_config


def main():
    """Main training function."""
    # Configure system
    configure_system()
    
    # Load configuration
    config = get_config()
    
    # Load data
    print("Loading data...")
    data_loader = DataLoader()
    X = data_loader.get_features()
    y = data_loader.get_targets()
    
    # Initialize and fit preprocessor
    print("Preprocessing data...")
    preprocessor = Preprocessor()
    X_processed = preprocessor.fit_transform(X)
    input_dim = X_processed.shape[1]
    
    print(f"Input dimension: {input_dim}")
    print(f"Training samples: {X_processed.shape[0]}")
    
    # Initialize trainer
    trainer = ModelTrainer(
        model_dir=config.get('model_dir'),
        model_name=config.get('model_name'),
        preprocessor=preprocessor
    )
    
    # Train model
    history = trainer.train(
        X_processed=X_processed,
        y=y.values,
        input_dim=input_dim,
        epochs=config.get('epochs'),
        batch_size=config.get('batch_size'),
        validation_split=config.get('validation_split'),
        verbose=1,
        hidden_layers=config.get('hidden_layers'),
        activation=config.get('activation'),
        optimizer=config.get('optimizer'),
        loss=config.get('loss')
    )
    
    # Save model and preprocessor
    print("\nSaving model...")
    saved_paths = trainer.save_model()
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {saved_paths['model']}")
    print(f"Preprocessor saved to: {saved_paths['preprocessor']}")
    print(f"Metadata saved to: {saved_paths['metadata']}")
    
    return 0


if __name__ == "__main__":
    exit(main())

