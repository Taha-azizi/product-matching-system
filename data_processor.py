"""
Data Processing Module
======================

Handles loading, cleaning, and preprocessing of product data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles all data loading and preprocessing operations."""
    
    def __init__(self):
        """Initialize the data processor."""
        pass
    
    def load_and_clean_data(
        self, 
        internal_path: str, 
        external_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and clean both internal and external product datasets.
        
        Args:
            internal_path: Path to internal products CSV
            external_path: Path to external products CSV
            
        Returns:
            Tuple of (cleaned_internal_data, cleaned_external_data)
        """
        try:
            # Load datasets
            internal_data = pd.read_csv(internal_path)
            external_data = pd.read_csv(external_path)
            
            logger.info(f"Loaded internal data: {internal_data.shape}")
            logger.info(f"Loaded external data: {external_data.shape}")
            
            # Clean internal data
            internal_data = self._clean_internal_data(internal_data)
            
            # Clean external data
            external_data = self._clean_external_data(external_data)
            
            # Initialize matching column in external data
            external_data['MATCHED_VALUE'] = pd.NA
            
            return internal_data, external_data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _clean_internal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess internal product data.
        
        Args:
            df: Raw internal data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Fill missing NAME values with OCS_NAME
        df['NAME'] = df['NAME'].fillna(df['OCS_NAME'])
        
        # Handle specific data quality issues
        # Fix the specific case mentioned in the notebook
        mask = df['NAME'] == 'MeltsVanCkieCrmPrBit45g'
        if mask.any():
            df.loc[mask, 'NAME'] = 'MeltsVanCkieCrmProBites45g'
        
        # Remove noise patterns (like 'zzz')
        df['NAME'] = df['NAME'].str.replace('zzz', '', regex=False)
        df['LONG_NAME'] = df['LONG_NAME'].str.replace('zzz', '', regex=False)
        
        # Ensure string types
        df['NAME'] = df['NAME'].astype(str)
        df['LONG_NAME'] = df['LONG_NAME'].astype(str)
        
        logger.info(f"Cleaned internal data: {df.shape}")
        return df
    
    def _clean_external_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess external product data.
        
        Args:
            df: Raw external data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Ensure string types
        df['PRODUCT_NAME'] = df['PRODUCT_NAME'].astype(str)
        
        # Remove any obvious noise or formatting issues
        df['PRODUCT_NAME'] = df['PRODUCT_NAME'].str.strip()
        
        logger.info(f"Cleaned external data: {df.shape}")
        return df
    
    def validate_data_quality(
        self, 
        internal_data: pd.DataFrame, 
        external_data: pd.DataFrame
    ) -> None:
        """
        Perform data quality validation checks.
        
        Args:
            internal_data: Internal products DataFrame
            external_data: External products DataFrame
        """
        # Check for required columns
        required_internal_cols = ['NAME', 'LONG_NAME']
        required_external_cols = ['PRODUCT_NAME']
        
        for col in required_internal_cols:
            if col not in internal_data.columns:
                raise ValueError(f"Missing required column in internal data: {col}")
        
        for col in required_external_cols:
            if col not in external_data.columns:
                raise ValueError(f"Missing required column in external data: {col}")
        
        # Check for empty datasets
        if len(internal_data) == 0:
            raise ValueError("Internal dataset is empty")
        
        if len(external_data) == 0:
            raise ValueError("External dataset is empty")
        
        logger.info("Data quality validation passed")