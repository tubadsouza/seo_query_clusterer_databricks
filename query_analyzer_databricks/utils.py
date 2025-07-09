"""
Utility functions for Databricks-compatible file operations and data processing.
"""

import pandas as pd
import json
from typing import Union, Optional, Dict, Any
from pathlib import Path
import os


def read_csv_dbfs(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Read CSV file with DBFS compatibility.
    
    Args:
        file_path: Path to CSV file (supports /dbfs/ paths)
        **kwargs: Additional arguments to pass to pd.read_csv()
    
    Returns:
        DataFrame containing the CSV data
    """
    # Handle DBFS paths
    if file_path.startswith('/dbfs/'):
        # In Databricks, /dbfs/ is mounted and accessible
        pass
    elif file_path.startswith('dbfs:/'):
        # Convert dbfs:/ to /dbfs/ format
        file_path = file_path.replace('dbfs:/', '/dbfs/')
    
    return pd.read_csv(file_path, **kwargs)


def write_csv_dbfs(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """
    Write DataFrame to CSV with DBFS compatibility.
    
    Args:
        df: DataFrame to write
        file_path: Output path (supports /dbfs/ paths)
        **kwargs: Additional arguments to pass to df.to_csv()
    """
    # Handle DBFS paths
    if file_path.startswith('/dbfs/'):
        # In Databricks, /dbfs/ is mounted and accessible
        pass
    elif file_path.startswith('dbfs:/'):
        # Convert dbfs:/ to /dbfs/ format
        file_path = file_path.replace('dbfs:/', '/dbfs/')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    df.to_csv(file_path, index=False, **kwargs)


def read_json_dbfs(file_path: str) -> Dict[str, Any]:
    """
    Read JSON file with DBFS compatibility.
    
    Args:
        file_path: Path to JSON file (supports /dbfs/ paths)
    
    Returns:
        Dictionary containing the JSON data
    """
    # Handle DBFS paths
    if file_path.startswith('/dbfs/'):
        pass
    elif file_path.startswith('dbfs:/'):
        file_path = file_path.replace('dbfs:/', '/dbfs/')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json_dbfs(data: Dict[str, Any], file_path: str) -> None:
    """
    Write data to JSON file with DBFS compatibility.
    
    Args:
        data: Dictionary to write
        file_path: Output path (supports /dbfs/ paths)
    """
    # Handle DBFS paths
    if file_path.startswith('/dbfs/'):
        pass
    elif file_path.startswith('dbfs:/'):
        file_path = file_path.replace('dbfs:/', '/dbfs/')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clean_query(query: str) -> str:
    """
    Clean and normalize a query for analysis.
    
    Args:
        query: Raw query string
    
    Returns:
        Cleaned query string
    """
    import re
    # Remove extra whitespace and normalize
    query = re.sub(r'\s+', ' ', query.strip())
    return query.lower()


def detect_language(text: str) -> str:
    """
    Detect the language of a text string.
    
    Args:
        text: Text to analyze
    
    Returns:
        Language code (e.g., 'en', 'es', etc.) or 'unknown' if detection fails
    """
    try:
        from langdetect import detect
        from langdetect.lang_detect_exception import LangDetectException
        try:
            return detect(text)
        except LangDetectException:
            return 'unknown'
    except ImportError:
        # Fallback: assume English if langdetect not available
        return 'en'


def is_english(text: str) -> bool:
    """
    Check if text is in English.
    
    Args:
        text: Text to check
    
    Returns:
        True if text is in English, False otherwise
    """
    return detect_language(text) == 'en' 