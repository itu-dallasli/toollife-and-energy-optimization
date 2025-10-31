"""
Model training module.
"""

import os
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import json
import time

from .model_builder import ModelBuilder
from ..data.preprocessor import Preprocessor


class ModelTrainer:
    """Handles model training and saving."""
    
    def __init__(
        self,
        model_dir: str = "models",
        model_name: str = "mlp_model",
        preprocessor: Optional[Preprocessor] = None
    ):
        """
        Initialize ModelTrainer.
        
        Args:
            model_dir: Directory to save models
            model_name: Base name for model files
            preprocessor: Preprocessor instance (optional)
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model_name = model_name
        self.preprocessor = preprocessor
        self.model = None
        self.history = None
        
    def configure_tensorflow(self, num_threads: Optional[int] = None) -> None:
        """
        Configure TensorFlow threading and GPU settings.
        
        Args:
            num_threads: Number of threads to use (default: CPU count)
        """
        if num_threads is None:
            import multiprocessing
            try:
                num_threads = multiprocessing.cpu_count()
            except NotImplementedError:
                num_threads = 4
        
        # Set threading
        try:
            tf.config.threading.set_inter_op_parallelism_threads(num_threads)
            tf.config.threading.set_intra_op_parallelism_threads(num_threads)
        except Exception as e:
            print(f"TensorFlow threading configuration failed: {e}")
        
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU configuration failed: {e}")
    
    def train(
        self,
        X_processed: np.ndarray,
        y: np.ndarray,
        input_dim: int,
        epochs: int = 500,
        batch_size: int = 4,
        validation_split: float = 0.0,
        verbose: int = 1,
        callbacks: Optional[list] = None,
        **model_kwargs
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_processed: Preprocessed feature data
            y: Target values
            input_dim: Input dimension
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data to use for validation
            verbose: Verbosity level
            callbacks: List of Keras callbacks
            **model_kwargs: Additional arguments for model building
            
        Returns:
            Training history dictionary
        """
        # Configure TensorFlow
        self.configure_tensorflow()
        
        # Build model
        self.model = ModelBuilder.build_mlp_model(input_dim, **model_kwargs)
        
        # Train model
        print(f"Training MLP model...")
        start_time = time.time()
        
        self.history = self.model.fit(
            X_processed,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=callbacks or []
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Convert history to dict
        history_dict = {}
        if isinstance(self.history, tf.keras.callbacks.History):
            history_dict = {key: [float(v) for v in values] 
                          for key, values in self.history.history.items()}
        else:
            history_dict = dict(self.history.history) if hasattr(self.history, 'history') else {}
        
        history_dict['training_time'] = training_time
        
        return history_dict
    
    def save_model(self, include_preprocessor: bool = True) -> Dict[str, str]:
        """
        Save trained model and optionally preprocessor.
        
        Args:
            include_preprocessor: Whether to save preprocessor
            
        Returns:
            Dictionary with paths to saved files
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        saved_paths = {}
        
        # Save model
        model_path = self.model_dir / f"{self.model_name}.keras"
        self.model.save(str(model_path))
        saved_paths['model'] = str(model_path)
        
        # Save preprocessor if provided
        if include_preprocessor and self.preprocessor is not None:
            preprocessor_path = self.model_dir / f"{self.model_name}_preprocessor.joblib"
            self.preprocessor.save(str(preprocessor_path))
            saved_paths['preprocessor'] = str(preprocessor_path)
        
        # Save training metadata
        metadata_path = self.model_dir / f"{self.model_name}_metadata.json"
        history_dict = {}
        if self.history:
            if hasattr(self.history, 'history'):
                history_dict = {key: [float(v) for v in values] 
                              for key, values in self.history.history.items()}
        
        metadata = {
            'model_name': self.model_name,
            'training_history': history_dict,
            'input_shape': str(self.model.input_shape),
            'output_shape': str(self.model.output_shape)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        saved_paths['metadata'] = str(metadata_path)
        
        print(f"Model saved successfully to {model_path}")
        return saved_paths
    
    def load_model(self, model_path: Optional[str] = None) -> tf.keras.Model:
        """
        Load a saved model.
        
        Args:
            model_path: Path to model file (default: uses model_name)
            
        Returns:
            Loaded Keras model
        """
        if model_path is None:
            model_path = self.model_dir / f"{self.model_name}.keras"
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = tf.keras.models.load_model(str(model_path))
        return self.model

