"""
Model architecture definition.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from typing import Dict, Any, Optional


class ModelBuilder:
    """Builds MLP model architectures for manufacturing prediction."""
    
    @staticmethod
    def build_mlp_model(
        input_shape: int,
        hidden_layers: Optional[list] = None,
        activation: str = 'relu',
        optimizer: str = 'adam',
        loss: str = 'mse',
        metrics: Optional[list] = None
    ) -> tf.keras.Model:
        """
        Build an MLP model for regression.
        
        Args:
            input_shape: Dimension of input features
            hidden_layers: List of hidden layer sizes (default: [64, 32])
            activation: Activation function for hidden layers
            optimizer: Optimizer name
            loss: Loss function
            metrics: List of metrics to track
            
        Returns:
            Compiled Keras model
        """
        if hidden_layers is None:
            hidden_layers = [64, 32]
        if metrics is None:
            metrics = ['mae']
        
        model = Sequential()
        
        # First hidden layer with input shape
        model.add(Dense(hidden_layers[0], activation=activation, input_shape=(input_shape,)))
        
        # Additional hidden layers
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation=activation))
        
        # Output layer (2 outputs: T and E)
        model.add(Dense(2, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    @staticmethod
    def get_model_summary(model: tf.keras.Model) -> str:
        """
        Get string summary of model architecture.
        
        Args:
            model: Keras model
            
        Returns:
            Model summary as string
        """
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            model.summary()
        return f.getvalue()

