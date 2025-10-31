"""
Comprehensive CLI for Manufacturing Parameter Optimization System.

Supports training, prediction, optimization, and comparison across different conditions.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import time
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.system_config import configure_system
from src.config.config import get_config
from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.models.trainer import ModelTrainer
from src.inference.predictor import Predictor
from src.inference.optimizer import ParameterOptimizer


# Available manufacturing conditions
VALID_CONDITIONS = ['Dry', 'MQL', 'Hybrid', 'Cryo', 'NF-1', 'NF-2']
CONDITION_LABELS = {
    'Dry': 'Kuru İşleme',
    'MQL': 'MQL (Minimum Yağlama)',
    'Hybrid': 'Hibrit Yağlama',
    'Cryo': 'Kriyojenik Soğutma',
    'NF-1': 'Nanofluid 1',
    'NF-2': 'Nanofluid 2'
}


def get_model_paths(config, model_path=None, preprocessor_path=None):
    """Get model and preprocessor paths."""
    if model_path is None:
        model_path = str(Path(config.get('model_dir')) / f"{config.get('model_name')}.keras")
    
    if preprocessor_path is None:
        preprocessor_path = str(Path(config.get('model_dir')) / f"{config.get('model_name')}_preprocessor.joblib")
    
    return model_path, preprocessor_path


def load_predictor(config, model_path=None, preprocessor_path=None) -> Predictor:
    """Load predictor with error handling."""
    model_path, preprocessor_path = get_model_paths(config, model_path, preprocessor_path)
    
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        print("\nHint: Train the model first using: python cli.py train")
        sys.exit(1)
    
    if not Path(preprocessor_path).exists():
        print(f"Error: Preprocessor file not found: {preprocessor_path}")
        print("\nHint: Train the model first using: python cli.py train")
        sys.exit(1)
    
    try:
        predictor = Predictor(
            model_path=model_path,
            preprocessor_path=preprocessor_path
        )
        return predictor
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def print_prediction_result(Vc: float, fn: float, condition: str, T: float, E: float, 
                          elapsed_time: float = None, show_label: bool = True):
    """Print formatted prediction result."""
    if show_label:
        label = CONDITION_LABELS.get(condition, condition)
        print(f"\n{'='*70}")
        print(f"Condition: {condition} ({label})")
        print(f"{'='*70}")
    
    print(f"Input Parameters:")
    print(f"  Cutting Speed (Vc): {Vc:.2f} m/min")
    print(f"  Feed Rate (fn):     {fn:.3f} mm/rev")
    print(f"\nPredictions:")
    print(f"  Tool Life (T):      {T:.2f} s")
    print(f"  Energy (E):         {E:.2f} kJ")
    print(f"  Efficiency (E/T):   {E/T:.5f}")
    
    if elapsed_time is not None:
        print(f"\n  Prediction Time:    {elapsed_time*1000:.4f} ms")


def cmd_train(args):
    """Train a new model."""
    print("="*70)
    print("Training Model")
    print("="*70)
    
    # Configure system
    configure_system()
    
    # Load configuration
    config = get_config()
    
    # Load data
    print("\n[1/4] Loading data...")
    data_loader = DataLoader()
    X = data_loader.get_features()
    y = data_loader.get_targets()
    print(f"  Loaded {len(X)} samples")
    print(f"  Features: {list(X.columns)}")
    print(f"  Targets: {list(y.columns)}")
    
    # Preprocess
    print("\n[2/4] Preprocessing data...")
    preprocessor = Preprocessor()
    X_processed = preprocessor.fit_transform(X)
    input_dim = X_processed.shape[1]
    print(f"  Input dimension: {input_dim}")
    print(f"  Training samples: {X_processed.shape[0]}")
    
    # Train
    print("\n[3/4] Training model...")
    trainer = ModelTrainer(
        model_dir=config.get('model_dir'),
        model_name=config.get('model_name'),
        preprocessor=preprocessor
    )
    
    history = trainer.train(
        X_processed=X_processed,
        y=y.values,
        input_dim=input_dim,
        epochs=config.get('epochs'),
        batch_size=config.get('batch_size'),
        validation_split=config.get('validation_split'),
        verbose=1 if args.verbose else 0,
        hidden_layers=config.get('hidden_layers'),
        activation=config.get('activation'),
        optimizer=config.get('optimizer'),
        loss=config.get('loss')
    )
    
    # Save
    print("\n[4/4] Saving model...")
    saved_paths = trainer.save_model()
    
    print("\n" + "="*70)
    print("Training Completed Successfully!")
    print("="*70)
    print(f"Model:        {saved_paths['model']}")
    print(f"Preprocessor: {saved_paths['preprocessor']}")
    print(f"Metadata:     {saved_paths['metadata']}")
    
    return 0


def cmd_predict(args):
    """Make predictions."""
    print("="*70)
    print("Making Prediction")
    print("="*70)
    
    # Configure system
    configure_system()
    
    # Load configuration
    config = get_config()
    
    # Validate condition
    if args.condition not in VALID_CONDITIONS:
        print(f"Error: Invalid condition '{args.condition}'")
        print(f"Valid conditions: {', '.join(VALID_CONDITIONS)}")
        return 1
    
    # Load predictor
    predictor = load_predictor(config, args.model_path, args.preprocessor_path)
    
    # Make prediction
    start_time = time.perf_counter()
    T, E = predictor.predict(args.Vc, args.fn, args.condition)
    elapsed_time = time.perf_counter() - start_time
    
    # Print results
    print_prediction_result(args.Vc, args.fn, args.condition, T, E, elapsed_time)
    
    # Output to file if requested
    if args.output:
        output_data = {
            'Vc': args.Vc,
            'fn': args.fn,
            'condition': args.condition,
            'T': float(T),
            'E': float(E),
            'efficiency': float(E/T)
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return 0


def cmd_predict_batch(args):
    """Make batch predictions from CSV file."""
    print("="*70)
    print("Batch Prediction")
    print("="*70)
    
    # Configure system
    configure_system()
    
    # Load configuration
    config = get_config()
    
    # Load input CSV
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    try:
        df = pd.read_csv(args.input)
        required_cols = ['Vc', 'fn', 'Condition']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: CSV must contain columns: {', '.join(required_cols)}")
            return 1
        
        # Validate conditions
        invalid_conditions = set(df['Condition']) - set(VALID_CONDITIONS)
        if invalid_conditions:
            print(f"Error: Invalid conditions found: {', '.join(invalid_conditions)}")
            print(f"Valid conditions: {', '.join(VALID_CONDITIONS)}")
            return 1
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return 1
    
    print(f"\nLoaded {len(df)} samples from {args.input}")
    
    # Load predictor
    predictor = load_predictor(config, args.model_path, args.preprocessor_path)
    
    # Make predictions
    print("\nMaking predictions...")
    start_time = time.perf_counter()
    predictions = predictor.predict_batch(df[required_cols])
    elapsed_time = time.perf_counter() - start_time
    
    # Add predictions to dataframe
    df['T_predicted'] = predictions[:, 0]
    df['E_predicted'] = predictions[:, 1]
    df['Efficiency_predicted'] = df['E_predicted'] / df['T_predicted']
    
    # Save results
    output_path = args.output if args.output else args.input.replace('.csv', '_predictions.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\nPredictions completed in {elapsed_time:.4f} seconds")
    print(f"Results saved to: {output_path}")
    
    # Show summary
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    print(df[['Condition', 'T_predicted', 'E_predicted', 'Efficiency_predicted']].groupby('Condition').agg({
        'T_predicted': ['mean', 'min', 'max'],
        'E_predicted': ['mean', 'min', 'max'],
        'Efficiency_predicted': ['mean', 'min', 'max']
    }).round(4))
    
    return 0


def cmd_optimize(args):
    """Optimize parameters for condition(s)."""
    print("="*70)
    print("Parameter Optimization")
    print("="*70)
    
    # Configure system
    configure_system()
    
    # Load configuration
    config = get_config()
    
    # Load predictor
    predictor = load_predictor(config, args.model_path, args.preprocessor_path)
    optimizer = ParameterOptimizer(predictor)
    
    # Determine conditions to optimize
    if args.all_conditions:
        conditions = VALID_CONDITIONS
        print("\nOptimizing for all conditions...")
    else:
        if args.condition not in VALID_CONDITIONS:
            print(f"Error: Invalid condition '{args.condition}'")
            print(f"Valid conditions: {', '.join(VALID_CONDITIONS)}")
            return 1
        conditions = [args.condition]
    
    # Get bounds
    Vc_bounds = args.Vc_bounds if args.Vc_bounds else config.get('Vc_bounds')
    fn_bounds = args.fn_bounds if args.fn_bounds else config.get('fn_bounds')
    
    # Optimize for each condition
    results = []
    for condition in conditions:
        print(f"\n{'-'*70}")
        print(f"Optimizing for condition: {condition} ({CONDITION_LABELS.get(condition, condition)})")
        print(f"{'-'*70}")
        
        start_time = time.perf_counter()
        result = optimizer.optimize(
            condition=condition,
            Vc_bounds=Vc_bounds,
            fn_bounds=fn_bounds,
            method=args.method if args.method else config.get('optimization_method'),
            maxiter=args.max_iterations if args.max_iterations else config.get('max_iterations'),
            minimize_ratio=not args.minimize_energy
        )
        elapsed_time = time.perf_counter() - start_time
        
        if result and result.get('success'):
            print(f"\n✓ Optimization Successful")
            print(f"  Optimal Vc:           {result['Vc']:.2f} m/min")
            print(f"  Optimal fn:           {result['fn']:.3f} mm/rev")
            print(f"  Predicted T:          {result['T']:.2f} s")
            print(f"  Predicted E:          {result['E']:.2f} kJ")
            print(f"  Efficiency (E/T):     {result['ratio']:.5f}")
            print(f"  Iterations:           {result['iterations']}")
            print(f"  Optimization Time:    {elapsed_time:.2f} s")
            result['condition'] = condition
            result['elapsed_time'] = elapsed_time
            results.append(result)
        else:
            print(f"\n✗ Optimization Failed")
            if result:
                print(f"  Message: {result.get('message', 'Unknown error')}")
                print(f"  Iterations: {result.get('iterations', 'N/A')}")
    
    # Save results if requested
    if args.output and results:
        output_data = []
        for r in results:
            output_data.append({
                'condition': r['condition'],
                'Vc': r['Vc'],
                'fn': r['fn'],
                'T': r['T'],
                'E': r['E'],
                'efficiency': r['ratio'],
                'iterations': r['iterations'],
                'optimization_time': r['elapsed_time']
            })
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Print comparison if optimizing for multiple conditions
    if len(results) > 1:
        print("\n" + "="*70)
        print("Comparison Across Conditions")
        print("="*70)
        print(f"{'Condition':<12} {'Vc':<8} {'fn':<8} {'T':<8} {'E':<8} {'E/T':<10}")
        print("-"*70)
        for r in results:
            print(f"{r['condition']:<12} {r['Vc']:>7.2f} {r['fn']:>7.3f} {r['T']:>7.2f} {r['E']:>7.2f} {r['ratio']:>9.5f}")
        
        # Find best condition
        best = min(results, key=lambda x: x['ratio'])
        print("\n" + "-"*70)
        print(f"Best condition (lowest E/T): {best['condition']} ({CONDITION_LABELS.get(best['condition'], best['condition'])})")
        print(f"  Efficiency: {best['ratio']:.5f}")
    
    return 0


def cmd_compare(args):
    """Compare predictions across different conditions."""
    print("="*70)
    print("Condition Comparison")
    print("="*70)
    
    # Configure system
    configure_system()
    
    # Load configuration
    config = get_config()
    
    # Validate conditions
    if args.all_conditions:
        conditions = VALID_CONDITIONS
    else:
        conditions = args.conditions if args.conditions else [args.condition]
        invalid = set(conditions) - set(VALID_CONDITIONS)
        if invalid:
            print(f"Error: Invalid conditions: {', '.join(invalid)}")
            print(f"Valid conditions: {', '.join(VALID_CONDITIONS)}")
            return 1
    
    # Load predictor
    predictor = load_predictor(config, args.model_path, args.preprocessor_path)
    
    # Make predictions for each condition
    print(f"\nComparing conditions with Vc={args.Vc:.2f}, fn={args.fn:.3f}")
    print("="*70)
    
    results = []
    for condition in conditions:
        T, E = predictor.predict(args.Vc, args.fn, condition)
        results.append({
            'condition': condition,
            'label': CONDITION_LABELS.get(condition, condition),
            'Vc': args.Vc,
            'fn': args.fn,
            'T': T,
            'E': E,
            'efficiency': E/T
        })
    
    # Sort by efficiency
    results.sort(key=lambda x: x['efficiency'])
    
    # Print comparison table
    print(f"\n{'Condition':<20} {'Label':<25} {'T (s)':<10} {'E (kJ)':<10} {'E/T':<12}")
    print("-"*70)
    for r in results:
        print(f"{r['condition']:<20} {r['label']:<25} {r['T']:>9.2f} {r['E']:>9.2f} {r['efficiency']:>11.5f}")
    
    print("\n" + "-"*70)
    best = results[0]
    worst = results[-1]
    print(f"Best (lowest E/T):  {best['condition']:<15} E/T = {best['efficiency']:.5f}")
    print(f"Worst (highest E/T): {worst['condition']:<15} E/T = {worst['efficiency']:.5f}")
    
    # Save results if requested
    if args.output:
        output_data = {
            'parameters': {'Vc': args.Vc, 'fn': args.fn},
            'results': results
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return 0


def cmd_list_conditions(args):
    """List available manufacturing conditions."""
    print("="*70)
    print("Available Manufacturing Conditions")
    print("="*70)
    
    data_loader = DataLoader()
    
    print(f"\n{'Code':<12} {'Label':<30} {'Samples':<10}")
    print("-"*70)
    for condition in VALID_CONDITIONS:
        label = CONDITION_LABELS.get(condition, condition)
        filtered = data_loader.filter_by_condition(condition)
        count = len(filtered)
        print(f"{condition:<12} {label:<30} {count:>9}")
    
    print(f"\nTotal conditions: {len(VALID_CONDITIONS)}")
    
    return 0


def cmd_info(args):
    """Show model information."""
    print("="*70)
    print("Model Information")
    print("="*70)
    
    # Configure system
    configure_system()
    
    # Load configuration
    config = get_config()
    
    model_path, preprocessor_path = get_model_paths(config, args.model_path, args.preprocessor_path)
    
    print(f"\nConfiguration:")
    print(f"  Model Directory:    {config.get('model_dir')}")
    print(f"  Model Name:         {config.get('model_name')}")
    print(f"  Model Path:         {model_path}")
    print(f"  Preprocessor Path:  {preprocessor_path}")
    
    # Check if model exists
    model_exists = Path(model_path).exists()
    preprocessor_exists = Path(preprocessor_path).exists()
    
    print(f"\nStatus:")
    print(f"  Model:              {'✓ Found' if model_exists else '✗ Not found'}")
    print(f"  Preprocessor:       {'✓ Found' if preprocessor_exists else '✗ Not found'}")
    
    if model_exists and preprocessor_exists:
        try:
            predictor = Predictor(model_path=model_path, preprocessor_path=preprocessor_path)
            print(f"  Model Ready:        ✓ Yes")
            
            # Try to get model info
            if predictor.model:
                print(f"\nModel Architecture:")
                print(f"  Input Shape:       {predictor.model.input_shape}")
                print(f"  Output Shape:      {predictor.model.output_shape}")
                print(f"  Parameters:        {predictor.model.count_params():,}")
                
                # Load metadata if available
                metadata_path = Path(config.get('model_dir')) / f"{config.get('model_name')}_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    print(f"\nTraining Information:")
                    for key, value in metadata.items():
                        if key not in ['model_path', 'preprocessor_path']:
                            print(f"  {key.capitalize():<20} {value}")
        except Exception as e:
            print(f"  Error loading model: {e}")
    
    print(f"\nAvailable Conditions: {', '.join(VALID_CONDITIONS)}")
    print(f"\nOptimization Bounds:")
    print(f"  Vc: {config.get('Vc_bounds')}")
    print(f"  fn: {config.get('fn_bounds')}")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Manufacturing Parameter Optimization System - CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python cli.py train
  
  # Make a prediction
  python cli.py predict --Vc 125.0 --fn 0.125 --condition Dry
  
  # Optimize for a condition
  python cli.py optimize --condition Dry
  
  # Compare all conditions
  python cli.py compare --Vc 125.0 --fn 0.125 --all
  
  # List available conditions
  python cli.py list-conditions
        """
    )
    
    parser.add_argument('--config', type=str, help='Path to config file')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--verbose', action='store_true', help='Verbose training output')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make a prediction')
    predict_parser.add_argument('--Vc', type=float, required=True, help='Cutting speed (m/min)')
    predict_parser.add_argument('--fn', type=float, required=True, help='Feed rate (mm/rev)')
    predict_parser.add_argument('--condition', type=str, required=True, choices=VALID_CONDITIONS, help='Manufacturing condition')
    predict_parser.add_argument('--model-path', type=str, help='Path to model file')
    predict_parser.add_argument('--preprocessor-path', type=str, help='Path to preprocessor file')
    predict_parser.add_argument('--output', '-o', type=str, help='Output JSON file')
    
    # Batch predict command
    batch_parser = subparsers.add_parser('predict-batch', help='Make batch predictions from CSV')
    batch_parser.add_argument('--input', '-i', type=str, required=True, help='Input CSV file with Vc, fn, Condition columns')
    batch_parser.add_argument('--output', '-o', type=str, help='Output CSV file (default: input_predictions.csv)')
    batch_parser.add_argument('--model-path', type=str, help='Path to model file')
    batch_parser.add_argument('--preprocessor-path', type=str, help='Path to preprocessor file')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize parameters for condition(s)')
    optimize_group = optimize_parser.add_mutually_exclusive_group(required=True)
    optimize_group.add_argument('--condition', type=str, choices=VALID_CONDITIONS, help='Manufacturing condition')
    optimize_group.add_argument('--all', '--all-conditions', dest='all_conditions', action='store_true', help='Optimize for all conditions')
    optimize_parser.add_argument('--Vc-bounds', nargs=2, type=float, metavar=('MIN', 'MAX'), help='Vc bounds (default: from config)')
    optimize_parser.add_argument('--fn-bounds', nargs=2, type=float, metavar=('MIN', 'MAX'), help='fn bounds (default: from config)')
    optimize_parser.add_argument('--method', type=str, help='Optimization method (default: COBYLA)')
    optimize_parser.add_argument('--max-iterations', type=int, help='Maximum iterations')
    optimize_parser.add_argument('--minimize-energy', action='store_true', help='Minimize energy instead of E/T ratio')
    optimize_parser.add_argument('--model-path', type=str, help='Path to model file')
    optimize_parser.add_argument('--preprocessor-path', type=str, help='Path to preprocessor file')
    optimize_parser.add_argument('--output', '-o', type=str, help='Output JSON file')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare predictions across conditions')
    compare_parser.add_argument('--Vc', type=float, required=True, help='Cutting speed (m/min)')
    compare_parser.add_argument('--fn', type=float, required=True, help='Feed rate (mm/rev)')
    compare_group = compare_parser.add_mutually_exclusive_group()
    compare_group.add_argument('--condition', type=str, choices=VALID_CONDITIONS, help='Single condition')
    compare_group.add_argument('--conditions', nargs='+', choices=VALID_CONDITIONS, help='Multiple conditions')
    compare_group.add_argument('--all', '--all-conditions', dest='all_conditions', action='store_true', help='All conditions')
    compare_parser.add_argument('--model-path', type=str, help='Path to model file')
    compare_parser.add_argument('--preprocessor-path', type=str, help='Path to preprocessor file')
    compare_parser.add_argument('--output', '-o', type=str, help='Output JSON file')
    
    # List conditions command
    list_parser = subparsers.add_parser('list-conditions', help='List available manufacturing conditions')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show model information')
    info_parser.add_argument('--model-path', type=str, help='Path to model file')
    info_parser.add_argument('--preprocessor-path', type=str, help='Path to preprocessor file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command handler
    handlers = {
        'train': cmd_train,
        'predict': cmd_predict,
        'predict-batch': cmd_predict_batch,
        'optimize': cmd_optimize,
        'compare': cmd_compare,
        'list-conditions': cmd_list_conditions,
        'info': cmd_info
    }
    
    handler = handlers.get(args.command)
    if handler:
        try:
            return handler(args)
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            return 130
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            import traceback
            if hasattr(args, 'verbose') and args.verbose:
                traceback.print_exc()
            return 1
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

