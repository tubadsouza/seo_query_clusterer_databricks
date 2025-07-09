"""
Main pipeline orchestration for the Query Analyzer (Databricks-optimized).
"""

import pandas as pd
from typing import Optional, Dict, Any, Tuple
from .data_loading import load_and_filter_queries, load_queries_from_table
from .embedding import embed_queries, add_embeddings_to_df
from .clustering import cluster_queries, generate_cluster_report
from .seo_generation import generate_seo_queries

from pyspark.sql import SparkSession

def run_pipeline(
    input_table: str,
    output_table: str,
    limit: int = 50000,
    # Data loading parameters
    min_length: int = 30,
    require_topics: bool = True,
    language_filter: bool = True,
    # Embedding parameters
    batch_size: int = 100,
    embedding_model: str = "text-embedding-3-large",
    api_key: Optional[str] = None,
    rate_limit_delay: float = 1.0,
    # Clustering parameters
    min_cluster_size: int = 3,
    max_cluster_size: int = 5,
    min_cluster_similarity: float = 0.6,
    min_seo_similarity: float = 0.8,
    min_unique_users: int = 3,
    # SEO generation parameters
    queries_per_cluster: int = 1
) -> Dict[str, Any]:
    """
    Run the complete query analysis pipeline (Databricks-optimized).
    """
    print("=" * 60)
    print("QUERY ANALYZER PIPELINE (Databricks)")
    print("=" * 60)
    # Step 1: Load and filter queries from Databricks table
    print("\nStep 1: Loading and filtering queries from table...")
    filtered_df = load_and_filter_queries(
        table_name=input_table,
        limit=limit,
        min_length=min_length,
        require_topics=require_topics,
        language_filter=language_filter
    )
    if len(filtered_df) == 0:
        raise ValueError("No queries remain after filtering")
    # Reset index to ensure alignment with embeddings
    filtered_df = filtered_df.reset_index(drop=True)
    # Step 2: Generate embeddings
    print("\nStep 2: Generating embeddings...")
    queries_list = filtered_df['query'].tolist()
    embeddings = embed_queries(
        queries=queries_list,
        batch_size=batch_size,
        model=embedding_model,
        api_key=api_key,
        rate_limit_delay=rate_limit_delay
    )
    # Add embeddings to DataFrame
    embedded_df = add_embeddings_to_df(filtered_df, embeddings)
    # (Optional) File-based embedding storage could be refactored to Delta in future
    # Step 3: Cluster queries
    print("\nStep 3: Clustering queries...")
    # Ensure alignment before clustering
    assert len(embeddings) == len(filtered_df), f"Embeddings ({len(embeddings)}) and filtered_df ({len(filtered_df)}) are not aligned!"
    # NOTE: Clustering is still done in Pandas/Numpy; could be migrated to Spark for large scale
    clustered_df, cluster_metadata = cluster_queries(
        embeddings=embeddings,
        queries_df=filtered_df,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        min_cluster_similarity=min_cluster_similarity,
        min_seo_similarity=min_seo_similarity,
        min_unique_users=min_unique_users
    )
    # Step 4: Generate SEO queries
    print("\nStep 4: Generating SEO queries...")
    seo_df = generate_seo_queries(
        clustered_df=clustered_df,
        queries_per_cluster=queries_per_cluster,
        api_key=api_key,
        rate_limit_delay=rate_limit_delay
    )
    # Write final SEO output to Databricks table (Delta)
    spark = SparkSession.builder.getOrCreate()
    spark_df = spark.createDataFrame(seo_df)
    spark_df.write.mode("overwrite").saveAsTable(output_table)
    print(f"Saved SEO queries to Databricks table: {output_table}")
    # Compile results
    results = {
        'filtered_queries': filtered_df,
        'clustered_queries': clustered_df,
        'seo_queries': seo_df,
        'cluster_metadata': cluster_metadata,
        'statistics': {
            'total_queries': len(filtered_df),
            'valid_clusters': len([m for m in cluster_metadata if not m['cluster_label'].endswith('_-1')]),
            'seo_candidates': len([m for m in cluster_metadata if m['seo_candidate'] == 'true']),
            'total_seo_queries': len(seo_df)
        }
    }
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total queries processed: {results['statistics']['total_queries']}")
    print(f"Valid clusters created: {results['statistics']['valid_clusters']}")
    print(f"SEO candidates identified: {results['statistics']['seo_candidates']}")
    print(f"SEO queries generated: {results['statistics']['total_seo_queries']}")
    print(f"Output table: {output_table}")
    return results


def run_pipeline_from_notebook(
    input_df: pd.DataFrame,
    output_dir: str = "/dbfs/query_analyzer/output",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for running pipeline from Databricks notebook.
    
    Args:
        input_df: Input DataFrame with queries
        output_dir: Output directory
        **kwargs: Additional pipeline parameters
    
    Returns:
        Pipeline results dictionary
    """
    return run_pipeline(
        input_df=input_df,
        output_dir=output_dir,
        **kwargs
    )


def get_pipeline_config() -> Dict[str, Any]:
    """
    Get default pipeline configuration.
    
    Returns:
        Dictionary with default configuration
    """
    return {
        'data_loading': {
            'min_length': 30,
            'require_topics': True,
            'language_filter': True
        },
        'embedding': {
            'batch_size': 100,
            'model': 'text-embedding-3-large',
            'rate_limit_delay': 1.0
        },
        'clustering': {
            'min_cluster_size': 3,
            'max_cluster_size': 5,
            'min_cluster_similarity': 0.6,
            'min_seo_similarity': 0.8,
            'min_unique_users': 3
        },
        'seo_generation': {
            'queries_per_cluster': 1
        }
    } 