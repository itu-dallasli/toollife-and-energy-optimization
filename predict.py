"""
Model prediction entry point.

This script loads a trained model and makes predictions.
Can be used for batch prediction or single predictions.
"""

import sys
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.system_config import configure_system
from src.config.config import get_config
from src.inference.predictor import Predictor
from src.inference.optimizer import ParameterOptimizer


def predict_single(predictor: Predictor, Vc: float, fn: float, condition: str):
    """Make a single prediction."""
    start_time = time.perf_counter()
    T, E = predictor.predict(Vc, fn, condition)
    elapsed_time = time.perf_counter() - start_time
    print(f"\nPrediction for Condition: {condition}, Vc={Vc:.2f}, fn={fn:.3f}")
    print(f"  Tool Life (T): {T:.2f} s")
    print(f"  Energy (E): {E:.2f} kJ")
    print(f"  Efficiency Ratio (E/T): {E/T:.5f}")
    print(f"  Prediction Time: {elapsed_time*1000:.4f} ms")
    return T, E


def optimize_parameters(optimizer: ParameterOptimizer, condition: str, config):
    """Optimize parameters for a condition."""
    print(f"\nOptimizing parameters for condition: {condition}...")
    result = optimizer.optimize(
        condition=condition,
        Vc_bounds=config.get('Vc_bounds'),
        fn_bounds=config.get('fn_bounds'),
        method=config.get('optimization_method'),
        maxiter=config.get('max_iterations')
    )
    
    if result['success']:
        print(f"\nOptimization Results:")
        print(f"  Optimal Vc: {result['Vc']:.2f} m/min")
        print(f"  Optimal fn: {result['fn']:.3f} mm/rev")
        print(f"  Predicted T: {result['T']:.2f} s")
        print(f"  Predicted E: {result['E']:.2f} kJ")
        print(f"  Efficiency Ratio (E/T): {result['ratio']:.5f}")
    else:
        print(f"Optimization failed: {result['message']}")
    
    return result


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model file (default: uses config)')
    parser.add_argument('--preprocessor-path', type=str, default=None,
                       help='Path to preprocessor file (default: uses config)')
    parser.add_argument('--Vc', type=float, default=125.0,
                       help='Cutting speed (m/min)')
    parser.add_argument('--fn', type=float, default=0.125,
                       help='Feed rate (mm/rev)')
    parser.add_argument('--condition', type=str, default='Dry',
                       choices=['Dry', 'MQL', 'Hybrid', 'Cryo', 'NF-1', 'NF-2'],
                       help='Manufacturing condition')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize parameters instead of predicting')
    
    args = parser.parse_args()
    
    # Configure system
    configure_system()
    
    # Load configuration
    config = get_config()
    
    # Determine model paths
    if args.model_path is None:
        args.model_path = str(Path(config.get('model_dir')) / f"{config.get('model_name')}.keras")
    
    if args.preprocessor_path is None:
        args.preprocessor_path = str(Path(config.get('model_dir')) / f"{config.get('model_name')}_preprocessor.joblib")
    
    # Load predictor
    print(f"Loading model from {args.model_path}...")
    try:
        predictor = Predictor(
            model_path=args.model_path,
            preprocessor_path=args.preprocessor_path
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nHint: Train the model first using: python train.py")
        return 1
    
    if args.optimize:
        # Optimize parameters
        optimizer = ParameterOptimizer(predictor)
        optimize_parameters(optimizer, args.condition, config)
    else:
        # Make prediction
        predict_single(predictor, args.Vc, args.fn, args.condition)
    
    return 0


if __name__ == "__main__":
    exit(main())

