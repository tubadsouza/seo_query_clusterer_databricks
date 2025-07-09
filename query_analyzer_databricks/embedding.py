"""
Embedding functions for query processing using OpenAI API.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
import time
import os
from tqdm import tqdm


def embed_queries(
    queries: List[str],
    batch_size: int = 100,
    model: str = "text-embedding-3-large",
    api_key: Optional[str] = None,
    rate_limit_delay: float = 1.0
) -> np.ndarray:
    """
    Generate embeddings for a list of queries using OpenAI API.
    
    Args:
        queries: List of query strings to embed
        batch_size: Number of queries to process in each batch
        model: OpenAI embedding model to use
        api_key: OpenAI API key (if None, uses environment variable)
        rate_limit_delay: Delay between batches in seconds
    
    Returns:
        numpy array of embeddings with shape (n_queries, embedding_dim)
        
    Raises:
        ValueError: If no API key is provided and OPENAI_API_KEY not set
        Exception: If API call fails
    """
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set")
    
    try:
        import openai
        openai.api_key = api_key
    except ImportError:
        raise ImportError("openai package not installed. Install with: pip install openai")
    
    if not queries:
        return np.array([])
    
    # Process in batches
    all_embeddings = []
    total_batches = (len(queries) + batch_size - 1) // batch_size
    
    print(f"Generating embeddings for {len(queries)} queries in {total_batches} batches...")
    
    for i in tqdm(range(0, len(queries), batch_size), desc="Embedding queries"):
        batch_queries = queries[i:i + batch_size]
        
        try:
            response = openai.embeddings.create(
                model=model,
                input=batch_queries
            )
            
            # Extract embeddings
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            # Rate limiting
            if i + batch_size < len(queries):  # Don't delay after last batch
                time.sleep(rate_limit_delay)
                
        except Exception as e:
            print(f"Error embedding batch {i//batch_size + 1}: {e}")
            # Return partial results if available
            if all_embeddings:
                print(f"Returning {len(all_embeddings)} embeddings from {len(queries)} queries")
                return np.array(all_embeddings)
            else:
                raise e
    
    embeddings_array = np.array(all_embeddings)
    print(f"Successfully generated {embeddings_array.shape[0]} embeddings with dimension {embeddings_array.shape[1]}")
    
    return embeddings_array


def add_embeddings_to_df(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_col: str = 'embedding'
) -> pd.DataFrame:
    """
    Add embeddings to a DataFrame.
    
    Args:
        df: DataFrame with queries
        embeddings: numpy array of embeddings
        embedding_col: Column name for embeddings
    
    Returns:
        DataFrame with embeddings added
    """
    if len(df) != len(embeddings):
        raise ValueError(f"DataFrame length ({len(df)}) doesn't match embeddings length ({len(embeddings)})")
    
    result_df = df.copy()
    result_df[embedding_col] = embeddings.tolist()
    return result_df


def validate_embeddings(embeddings: np.ndarray) -> dict:
    """
    Validate embedding array and return statistics.
    
    Args:
        embeddings: numpy array of embeddings
    
    Returns:
        Dictionary with validation statistics
    """
    if len(embeddings) == 0:
        return {'count': 0, 'dimension': 0, 'mean_norm': 0, 'std_norm': 0}
    
    # Calculate norms
    norms = np.linalg.norm(embeddings, axis=1)
    
    stats = {
        'count': len(embeddings),
        'dimension': embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
        'mean_norm': float(np.mean(norms)),
        'std_norm': float(np.std(norms)),
        'min_norm': float(np.min(norms)),
        'max_norm': float(np.max(norms))
    }
    
    return stats 