"""
Model prediction and inference module.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple, Optional
from pathlib import Path

from ..data.preprocessor import Preprocessor


class Predictor:
    """Handles model loading and prediction."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
        preprocessor: Optional[Preprocessor] = None
    ):
        """
        Initialize Predictor.
        
        Args:
            model_path: Path to saved model file
            preprocessor_path: Path to saved preprocessor file
            preprocessor: Preprocessor instance (optional, overrides preprocessor_path)
        """
        self.model = None
        self.preprocessor = preprocessor
        
        if model_path:
            self.load_model(model_path)
        
        if preprocessor and not self.preprocessor:
            self.preprocessor = preprocessor
        elif preprocessor_path and not self.preprocessor:
            self.load_preprocessor(preprocessor_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load model from file.
        
        Args:
            model_path: Path to model file
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = tf.keras.models.load_model(str(path))
        print(f"Model loaded successfully from {model_path}")
    
    def load_preprocessor(self, preprocessor_path: str) -> None:
        """
        Load preprocessor from file.
        
        Args:
            preprocessor_path: Path to preprocessor file
        """
        path = Path(preprocessor_path)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
        
        self.preprocessor = Preprocessor()
        self.preprocessor.load(str(path))
        print(f"Preprocessor loaded successfully from {preprocessor_path}")
    
    def predict(
        self,
        Vc: float,
        fn: float,
        condition: str,
        return_raw: bool = False
    ) -> Tuple[float, float]:
        """
        Make prediction for given parameters.
        
        Args:
            Vc: Cutting speed (m/min)
            fn: Feed rate (mm/rev)
            condition: Manufacturing condition
            return_raw: If True, return raw numpy array instead of tuple
            
        Returns:
            Tuple of (T, E) predictions, or raw array if return_raw=True
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded. Call load_preprocessor() first.")
        
        # Prepare input data
        input_data = pd.DataFrame([[Vc, fn, condition]], columns=['Vc', 'fn', 'Condition'])
        
        # Transform
        input_processed = self.preprocessor.transform(input_data)
        
        # Predict
        prediction = self.model.predict(input_processed, verbose=0)[0]
        
        if return_raw:
            return prediction
        else:
            return float(prediction[0]), float(prediction[1])
    
    def predict_batch(
        self,
        data: pd.DataFrame,
        return_raw: bool = False
    ) -> np.ndarray:
        """
        Make batch predictions.
        
        Args:
            data: DataFrame with columns ['Vc', 'fn', 'Condition']
            return_raw: If True, return raw predictions without unpacking
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded. Call load_preprocessor() first.")
        
        # Transform
        data_processed = self.preprocessor.transform(data)
        
        # Predict
        predictions = self.model.predict(data_processed, verbose=0)
        
        return predictions
    
    def is_ready(self) -> bool:
        """Check if model and preprocessor are loaded and ready."""
        return self.model is not None and self.preprocessor is not None

