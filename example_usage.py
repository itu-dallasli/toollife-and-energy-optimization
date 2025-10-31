"""
Example usage of the optimization system API.

This demonstrates how to use the system programmatically,
which is useful for integration into larger systems.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.system_config import configure_system
from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.models.trainer import ModelTrainer
from src.inference.predictor import Predictor
from src.inference.optimizer import ParameterOptimizer
from src.config.config import get_config


def example_training():
    """Example: Train a model."""
    print("=" * 60)
    print("Example: Training a Model")
    print("=" * 60)
    
    # Configure system
    configure_system()
    
    # Load configuration
    config = get_config()
    
    # Load data
    print("\n1. Loading data...")
    data_loader = DataLoader()
    X = data_loader.get_features()
    y = data_loader.get_targets()
    print(f"   Loaded {len(X)} samples")
    
    # Preprocess
    print("\n2. Preprocessing data...")
    preprocessor = Preprocessor()
    X_processed = preprocessor.fit_transform(X)
    print(f"   Input dimension: {X_processed.shape[1]}")
    
    # Train
    print("\n3. Training model...")
    trainer = ModelTrainer(
        model_dir=config.get('model_dir'),
        model_name=config.get('model_name'),
        preprocessor=preprocessor
    )
    
    history = trainer.train(
        X_processed=X_processed,
        y=y.values,
        input_dim=X_processed.shape[1],
        epochs=config.get('epochs'),
        batch_size=config.get('batch_size')
    )
    
    # Save
    print("\n4. Saving model...")
    saved_paths = trainer.save_model(include_preprocessor=True)
    print(f"   Model: {saved_paths['model']}")
    print(f"   Preprocessor: {saved_paths['preprocessor']}")
    
    return saved_paths


def example_prediction(model_path: str, preprocessor_path: str):
    """Example: Make predictions."""
    print("\n" + "=" * 60)
    print("Example: Making Predictions")
    print("=" * 60)
    
    # Load predictor
    print("\n1. Loading model...")
    predictor = Predictor(
        model_path=model_path,
        preprocessor_path=preprocessor_path
    )
    
    # Make predictions
    print("\n2. Making predictions...")
    test_cases = [
        (100, 0.10, 'Dry'),
        (125, 0.125, 'MQL'),
        (150, 0.15, 'Hybrid'),
    ]
    
    for Vc, fn, condition in test_cases:
        T, E = predictor.predict(Vc, fn, condition)
        print(f"\n   Condition: {condition}, Vc={Vc}, fn={fn}")
        print(f"   → Tool Life (T): {T:.2f} s")
        print(f"   → Energy (E): {E:.2f} kJ")
        print(f"   → Efficiency (E/T): {E/T:.5f}")


def example_optimization(model_path: str, preprocessor_path: str):
    """Example: Optimize parameters."""
    print("\n" + "=" * 60)
    print("Example: Parameter Optimization")
    print("=" * 60)
    
    # Load predictor
    print("\n1. Loading model...")
    predictor = Predictor(
        model_path=model_path,
        preprocessor_path=preprocessor_path
    )
    
    # Create optimizer
    optimizer = ParameterOptimizer(predictor)
    config = get_config()
    
    # Optimize for different conditions
    print("\n2. Optimizing parameters...")
    conditions = ['Dry', 'MQL', 'Hybrid']
    
    for condition in conditions:
        print(f"\n   Optimizing for {condition}...")
        result = optimizer.optimize(
            condition=condition,
            Vc_bounds=config.get('Vc_bounds'),
            fn_bounds=config.get('fn_bounds'),
            method=config.get('optimization_method'),
            maxiter=config.get('max_iterations')
        )
        
        if result['success']:
            print(f"   ✓ Optimal Vc: {result['Vc']:.2f} m/min")
            print(f"   ✓ Optimal fn: {result['fn']:.3f} mm/rev")
            print(f"   ✓ Predicted T: {result['T']:.2f} s")
            print(f"   ✓ Predicted E: {result['E']:.2f} kJ")
            print(f"   ✓ Efficiency: {result['ratio']:.5f}")
        else:
            print(f"   ✗ Optimization failed: {result['message']}")


def main():
    """Main example function."""
    print("\n" + "=" * 60)
    print("Optimization System - Example Usage")
    print("=" * 60)
    
    # Check if model exists
    config = get_config()
    model_path = Path(config.get('model_dir')) / f"{config.get('model_name')}.keras"
    preprocessor_path = Path(config.get('model_dir')) / f"{config.get('model_name')}_preprocessor.joblib"
    
    if not model_path.exists():
        print("\nModel not found. Training a new model first...")
        saved_paths = example_training()
        model_path = saved_paths['model']
        preprocessor_path = saved_paths['preprocessor']
    else:
        print(f"\nUsing existing model: {model_path}")
    
    # Run examples
    example_prediction(str(model_path), str(preprocessor_path))
    example_optimization(str(model_path), str(preprocessor_path))
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

