"""
Data loading and filtering functions for query processing (Databricks-optimized).
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from .utils import clean_query, is_english

from pyspark.sql import SparkSession

def load_queries_from_table(table_name: str, limit: int = 10000):
    """
    Load queries from a Databricks table using Spark SQL.
    """
    spark = SparkSession.builder.getOrCreate()
    df = spark.sql(f"SELECT query, topic, user_id FROM {table_name} LIMIT {limit}")
    return df.toPandas()


def load_and_filter_queries(
    table_name: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    limit: int = 10000,
    min_length: int = 30,
    require_topics: bool = True,
    language_filter: bool = True
) -> pd.DataFrame:
    """
    Load and filter queries from Databricks table or DataFrame.
    """
    if df is None and table_name is None:
        raise ValueError("Either table_name or df must be provided")
    if df is None:
        df = load_queries_from_table(table_name, limit=limit)
    # Ensure required columns exist
    required_cols = ['query']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    original_count = len(df)
    filtered_df = df.copy()
    filters_applied = {}
    if min_length > 0:
        length_mask = filtered_df['query'].str.len() >= min_length
        filters_applied['length'] = original_count - length_mask.sum()
        filtered_df = filtered_df[length_mask]
    if language_filter:
        lang_mask = filtered_df['query'].apply(is_english)
        filters_applied['language'] = len(filtered_df) - lang_mask.sum()
        filtered_df = filtered_df[lang_mask]
    if require_topics and 'topic' in filtered_df.columns:
        topic_mask = filtered_df['topic'].notna() & (filtered_df['topic'] != '')
        filters_applied['topics'] = len(filtered_df) - topic_mask.sum()
        filtered_df = filtered_df[topic_mask]
    filtered_df['query_clean'] = filtered_df['query'].apply(clean_query)
    duplicate_mask = ~filtered_df['query_clean'].duplicated()
    filters_applied['duplicates'] = len(filtered_df) - duplicate_mask.sum()
    filtered_df = filtered_df[duplicate_mask]
    filtered_df['query'] = filtered_df['query'].apply(clean_query)
    result_df = pd.DataFrame({
        'query': filtered_df['query'],
        'user': filtered_df.get('user_id', filtered_df.get('user', 'unknown')),
        'topics': filtered_df.get('topic', '').fillna('')
    })
    if 'topics' in result_df.columns:
        result_df['topics'] = result_df['topics'].apply(
            lambda x: [t.strip() for t in str(x).split(',')] if pd.notna(x) and x != '' else []
        )
    final_count = len(result_df)
    print(f"Query filtering summary:")
    print(f"  Original queries: {original_count}")
    for filter_name, removed_count in filters_applied.items():
        print(f"  Removed by {filter_name}: {removed_count}")
    print(f"  Final queries: {final_count}")
    print(f"  Retention rate: {final_count/original_count:.1%}")
    return result_df

def validate_query_data(df: pd.DataFrame) -> Dict[str, Any]:
    stats = {
        'total_queries': len(df),
        'unique_queries': df['query'].nunique(),
        'unique_users': df['user'].nunique(),
        'avg_query_length': df['query'].str.len().mean(),
        'min_query_length': df['query'].str.len().min(),
        'max_query_length': df['query'].str.len().max(),
        'queries_with_topics': df['topics'].apply(len).gt(0).sum() if 'topics' in df.columns else 0
    }
    try:
        from .utils import detect_language
        df['language'] = df['query'].apply(detect_language)
        stats['language_distribution'] = df['language'].value_counts().to_dict()
    except:
        stats['language_distribution'] = {'unknown': len(df)}
    return stats 