"""
Parameter optimization module.
"""

from typing import Dict, Optional, Tuple
from scipy.optimize import minimize
import numpy as np

from .predictor import Predictor


class ParameterOptimizer:
    """Optimizes manufacturing parameters using trained model."""
    
    def __init__(self, predictor: Predictor):
        """
        Initialize ParameterOptimizer.
        
        Args:
            predictor: Predictor instance for making predictions
        """
        self.predictor = predictor
    
    def objective_function(
        self,
        params: np.ndarray,
        condition: str,
        minimize_ratio: bool = True
    ) -> float:
        """
        Objective function for optimization.
        
        Args:
            params: [Vc, fn] array
            condition: Manufacturing condition
            minimize_ratio: If True, minimize E/T ratio; if False, minimize E
            
        Returns:
            Objective value (E/T or E)
        """
        Vc, fn = params
        T, E = self.predictor.predict(Vc, fn, condition)
        
        # Handle invalid predictions
        if T <= 0 or E == float('inf') or np.isnan(E) or np.isnan(T):
            return 1e9
        
        if minimize_ratio:
            return E / T
        else:
            return E
    
    def optimize(
        self,
        condition: str,
        Vc_bounds: Tuple[float, float] = (80, 170),
        fn_bounds: Tuple[float, float] = (0.08, 0.17),
        initial_guess: Optional[Tuple[float, float]] = None,
        method: str = 'COBYLA',
        maxiter: int = 5000,
        minimize_ratio: bool = True
    ) -> Optional[Dict]:
        """
        Optimize parameters for given condition.
        
        Args:
            condition: Manufacturing condition
            Vc_bounds: (min, max) bounds for cutting speed
            fn_bounds: (min, max) bounds for feed rate
            initial_guess: Initial guess [Vc, fn] (default: middle of bounds)
            method: Optimization method
            maxiter: Maximum iterations
            minimize_ratio: If True, minimize E/T; if False, minimize E
            
        Returns:
            Dictionary with optimization results or None if failed
        """
        if initial_guess is None:
            initial_guess = [
                (Vc_bounds[0] + Vc_bounds[1]) / 2,
                (fn_bounds[0] + fn_bounds[1]) / 2
            ]
        
        bounds = [Vc_bounds, fn_bounds]
        
        result = minimize(
            self.objective_function,
            initial_guess,
            args=(condition, minimize_ratio),
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter}
        )
        
        if result.success:
            opt_Vc, opt_fn = result.x
            opt_T, opt_E = self.predictor.predict(opt_Vc, opt_fn, condition)
            
            return {
                'Vc': float(opt_Vc),
                'fn': float(opt_fn),
                'T': float(opt_T),
                'E': float(opt_E),
                'ratio': float(opt_E / opt_T) if opt_T > 0 else None,
                'success': True,
                'message': result.message,
                'iterations': result.nit
            }
        else:
            return {
                'success': False,
                'message': result.message,
                'iterations': result.nit
            }

