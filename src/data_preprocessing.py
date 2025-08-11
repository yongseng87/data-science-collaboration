"""
Data preprocessing module for the data science collaboration project.

This module contains functions for cleaning, transforming, and preparing
raw data for analysis and model training.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(data)} records from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def clean_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): Strategy for handling missing values ('drop', 'mean', 'median')
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    if strategy == 'drop':
        cleaned_df = df.dropna()
        logger.info(f"Dropped {len(df) - len(cleaned_df)} rows with missing values")
    elif strategy == 'mean':
        cleaned_df = df.fillna(df.mean(numeric_only=True))
        logger.info("Filled missing values with column means")
    elif strategy == 'median':
        cleaned_df = df.fillna(df.median(numeric_only=True))
        logger.info("Filled missing values with column medians")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return cleaned_df


def remove_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
    """
    Remove outliers from specified columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str]): Columns to check for outliers
        method (str): Method for outlier detection ('iqr', 'zscore')
        
    Returns:
        pd.DataFrame: Dataframe with outliers removed
    """
    cleaned_df = df.copy()
    
    for column in columns:
        if column not in df.columns:
            logger.warning(f"Column {column} not found in dataframe")
            continue
            
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outlier_mask = z_scores > 3
        else:
            raise ValueError(f"Unknown method: {method}")
            
        outlier_count = outlier_mask.sum()
        cleaned_df = cleaned_df[~outlier_mask]
        logger.info(f"Removed {outlier_count} outliers from column {column}")
    
    return cleaned_df


def normalize_features(df: pd.DataFrame, columns: List[str], method: str = 'standard') -> pd.DataFrame:
    """
    Normalize numerical features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str]): Columns to normalize
        method (str): Normalization method ('standard', 'minmax')
        
    Returns:
        pd.DataFrame: Dataframe with normalized features
    """
    normalized_df = df.copy()
    
    for column in columns:
        if column not in df.columns:
            logger.warning(f"Column {column} not found in dataframe")
            continue
            
        if method == 'standard':
            mean = df[column].mean()
            std = df[column].std()
            normalized_df[column] = (df[column] - mean) / std
        elif method == 'minmax':
            min_val = df[column].min()
            max_val = df[column].max()
            normalized_df[column] = (df[column] - min_val) / (max_val - min_val)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Normalized column {column} using {method} method")
    
    return normalized_df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing ones.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new features
    """
    enhanced_df = df.copy()
    
    # Example feature engineering
    if 'date' in df.columns:
        enhanced_df['date'] = pd.to_datetime(enhanced_df['date'])
        enhanced_df['year'] = enhanced_df['date'].dt.year
        enhanced_df['month'] = enhanced_df['date'].dt.month
        enhanced_df['day_of_week'] = enhanced_df['date'].dt.dayofweek
        logger.info("Created date-based features")
    
    # Add more feature engineering logic here
    
    return enhanced_df


def save_processed_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save processed data to a CSV file.
    
    Args:
        df (pd.DataFrame): Processed dataframe
        file_path (str): Path to save the file
    """
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Successfully saved {len(df)} records to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise


def preprocess_pipeline(raw_data_path: str, processed_data_path: str) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    
    Args:
        raw_data_path (str): Path to raw data file
        processed_data_path (str): Path to save processed data
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    logger.info("Starting data preprocessing pipeline...")
    
    # Load data
    df = load_raw_data(raw_data_path)
    
    # Clean missing values
    df = clean_missing_values(df, strategy='median')
    
    # Remove outliers (example columns)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        df = remove_outliers(df, numeric_columns[:2])  # Remove outliers from first 2 numeric columns
    
    # Normalize features
    if numeric_columns:
        df = normalize_features(df, numeric_columns, method='standard')
    
    # Create new features
    df = create_features(df)
    
    # Save processed data
    save_processed_data(df, processed_data_path)
    
    logger.info("Data preprocessing pipeline completed successfully!")
    return df


if __name__ == "__main__":
    # Example usage
    raw_path = "../data/raw/sample_data.csv"
    processed_path = "../data/processed/processed_data.csv"
    
    try:
        processed_data = preprocess_pipeline(raw_path, processed_path)
        print(f"Preprocessing completed. Shape: {processed_data.shape}")
    except Exception as e:
        print(f"Error in preprocessing pipeline: {str(e)}")
