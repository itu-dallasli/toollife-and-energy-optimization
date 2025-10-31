"""
Data loading module for manufacturing optimization data.
"""

import pandas as pd
from typing import List, Dict, Tuple
import os


class DataLoader:
    """Handles loading and management of manufacturing data."""
    
    # Manufacturing condition data
    DATA_LIST = [
        [100, 0.10, 'Dry', 60, 23.29], [100, 0.15, 'Dry', 48, 19.64],
        [150, 0.10, 'Dry', 35, 19.91], [150, 0.15, 'Dry', 32, 17.75],
        [100, 0.10, 'MQL', 72, 24.05], [100, 0.15, 'MQL', 68, 20.51],
        [150, 0.10, 'MQL', 63, 20.13], [150, 0.15, 'MQL', 57, 18.47],
        [100, 0.10, 'Hybrid', 94, 25.56], [100, 0.15, 'Hybrid', 89, 21.17],
        [150, 0.10, 'Hybrid', 75, 20.91], [150, 0.15, 'Hybrid', 66, 19.29],
        [100, 0.10, 'Cryo', 121, 22.63], [100, 0.15, 'Cryo', 162, 18.32],
        [150, 0.10, 'Cryo', 107, 18.51], [150, 0.15, 'Cryo', 101, 17.24],
        [100, 0.10, 'NF-1', 175, 24.42], [100, 0.15, 'NF-1', 157, 22.85],
        [150, 0.10, 'NF-1', 149, 22.46], [150, 0.15, 'NF-1', 103, 20.88],
        [100, 0.10, 'NF-2', 202, 23.26], [100, 0.15, 'NF-2', 200, 21.62],
        [150, 0.10, 'NF-2', 189, 21.44], [150, 0.15, 'NF-2', 170, 19.59],
    ]
    
    COND_LABELS = {
        'Dry': 'Kuru İşleme',
        'MQL': 'MQL (Minimum Yağlama)',
        'Hybrid': 'Hibrit Yağlama',
        'Cryo': 'Kriyojenik Soğutma',
        'NF-1': 'Nanofluid 1',
        'NF-2': 'Nanofluid 2'
    }
    
    COLUMNS = ['Vc', 'fn', 'Condition', 'T', 'E']
    
    def __init__(self):
        """Initialize DataLoader."""
        self.df = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load data from internal list."""
        self.df = pd.DataFrame(self.DATA_LIST, columns=self.COLUMNS)
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load data from external file.
        
        Args:
            filepath: Path to CSV file with columns matching self.COLUMNS
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        self.df = pd.read_csv(filepath)
        # Validate columns
        if list(self.df.columns) != self.COLUMNS:
            raise ValueError(f"Expected columns {self.COLUMNS}, got {list(self.df.columns)}")
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get the loaded dataframe."""
        return self.df.copy()
    
    def get_features(self) -> pd.DataFrame:
        """Get feature columns (Vc, fn, Condition)."""
        return self.df[['Vc', 'fn', 'Condition']].copy()
    
    def get_targets(self) -> pd.DataFrame:
        """Get target columns (T, E)."""
        return self.df[['T', 'E']].copy()
    
    def get_unique_conditions(self) -> List[str]:
        """Get list of unique manufacturing conditions."""
        return self.df['Condition'].unique().tolist()
    
    def get_condition_label(self, condition: str) -> str:
        """Get human-readable label for a condition."""
        return self.COND_LABELS.get(condition, condition)
    
    def filter_by_condition(self, condition: str) -> pd.DataFrame:
        """Filter dataframe by manufacturing condition."""
        return self.df[self.df['Condition'] == condition].copy()

