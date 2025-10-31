"""
Data preprocessing module for feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Tuple
import joblib
import os


class Preprocessor:
    """Handles data preprocessing and transformation."""
    
    def __init__(self):
        """Initialize Preprocessor."""
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['Vc', 'fn']),
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['Condition'])
            ],
            remainder='passthrough'
        )
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame) -> 'Preprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            X: DataFrame with columns ['Vc', 'fn', 'Condition']
            
        Returns:
            self
        """
        self.preprocessor.fit(X)
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: DataFrame with columns ['Vc', 'fn', 'Condition']
            
        Returns:
            Transformed numpy array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")
        return self.preprocessor.transform(X)
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit preprocessor and transform data.
        
        Args:
            X: DataFrame with columns ['Vc', 'fn', 'Condition']
            
        Returns:
            Transformed numpy array
        """
        result = self.preprocessor.fit_transform(X)
        self.is_fitted = True
        return result
    
    def get_input_dimension(self) -> int:
        """Get the dimension of preprocessed input features."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first.")
        # Create dummy data to get output shape
        dummy = pd.DataFrame([[100, 0.10, 'Dry']], columns=['Vc', 'fn', 'Condition'])
        return self.preprocessor.transform(dummy).shape[1]
    
    def save(self, filepath: str) -> None:
        """
        Save preprocessor to disk.
        
        Args:
            filepath: Path to save the preprocessor
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor.")
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        joblib.dump(self.preprocessor, filepath)
    
    def load(self, filepath: str) -> 'Preprocessor':
        """
        Load preprocessor from disk.
        
        Args:
            filepath: Path to load the preprocessor from
            
        Returns:
            self
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
        self.preprocessor = joblib.load(filepath)
        self.is_fitted = True
        return self

