"""
Clustering functions for query analysis using HDBSCAN.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def cluster_queries(
    embeddings: np.ndarray,
    queries_df: pd.DataFrame,
    min_cluster_size: int = 3,
    max_cluster_size: int = 5,
    min_cluster_similarity: float = 0.6,
    min_seo_similarity: float = 0.8,
    min_unique_users: int = 3,
    cluster_selection_epsilon: float = 0.1
) -> pd.DataFrame:
    """
    Cluster queries using HDBSCAN with topic-based grouping.
    
    Args:
        embeddings: numpy array of embeddings
        queries_df: DataFrame with queries, users, and topics
        min_cluster_size: Minimum queries per cluster
        max_cluster_size: Maximum queries per cluster
        min_cluster_similarity: Minimum similarity for valid clusters
        min_seo_similarity: Minimum similarity for SEO candidates
        min_unique_users: Minimum unique users per cluster
        cluster_selection_epsilon: HDBSCAN epsilon parameter
    
    Returns:
        DataFrame with cluster assignments and metadata
    """
    if len(embeddings) != len(queries_df):
        raise ValueError("Embeddings and queries_df must have the same length")
    
    # Group queries by topic
    topic_groups = defaultdict(list)
    for idx, row in queries_df.iterrows():
        topics = row.get('topics', [])
        if isinstance(topics, str):
            topics = [t.strip() for t in topics.split(',') if t.strip()]
        
        for topic in topics:
            topic_groups[topic].append({
                'index': idx,
                'query': row['query'],
                'user': row['user'],
                'topics': topics,
                'embedding': embeddings[idx]
            })
    
    print(f"Processing {len(topic_groups)} topics...")
    
    # Process each topic
    clustered_results = []
    cluster_metadata = []
    processed_queries = set()
    
    for topic, group in tqdm(topic_groups.items(), desc="Clustering by topic"):
        if len(group) < min_cluster_size:
            print(f"[DEBUG] Topic '{topic}' skipped: only {len(group)} queries (min required: {min_cluster_size})")
            # Assign small groups to noise
            for item in group:
                query_key = (item['query'], tuple(item['topics']))
                if query_key not in processed_queries:
                    clustered_results.append({
                        'query': item['query'],
                        'user': item['user'],
                        'topics': item['topics'],
                        'cluster': f"{topic}_-1",
                        'embedding': item['embedding']
                    })
                    processed_queries.add(query_key)
            continue
        
        # Prepare embeddings for clustering
        group_embeddings = np.array([item['embedding'] for item in group])
        
        try:
            # Cluster with HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=2,
                metric='euclidean',
                cluster_selection_epsilon=cluster_selection_epsilon,
                cluster_selection_method='leaf'
            )
            labels = clusterer.fit_predict(group_embeddings)
            
            # Process each cluster
            for label in set(labels):
                if label == -1:
                    continue
                
                # Get queries for this cluster
                cluster_items = [item for i, item in enumerate(group) if labels[i] == label]
                cluster_embeddings = group_embeddings[labels == label]
                
                # Limit cluster size
                if len(cluster_items) > max_cluster_size:
                    # Keep most similar queries
                    centroid = np.mean(cluster_embeddings, axis=0)
                    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                    closest_indices = np.argsort(distances)[:max_cluster_size]
                    cluster_items = [cluster_items[i] for i in closest_indices]
                    cluster_embeddings = cluster_embeddings[closest_indices]
                
                # Validate cluster
                is_valid, metadata = validate_cluster(
                    cluster_items, cluster_embeddings, topic, label,
                    min_cluster_similarity, min_seo_similarity, min_unique_users, min_cluster_size
                )
                
                if not is_valid:
                    print(f"[DEBUG] Cluster {topic}_{label} rejected. Reason(s):", metadata)
                
                if is_valid:
                    # Add cluster metadata
                    cluster_metadata.append(metadata)
                    
                    # Add clustered queries
                    for item in cluster_items:
                        query_key = (item['query'], tuple(item['topics']))
                        if query_key not in processed_queries:
                            clustered_results.append({
                                'query': item['query'],
                                'user': item['user'],
                                'topics': item['topics'],
                                'cluster': metadata['cluster_label'],
                                'embedding': item['embedding']
                            })
                            processed_queries.add(query_key)
                else:
                    # Assign to noise
                    for item in cluster_items:
                        query_key = (item['query'], tuple(item['topics']))
                        if query_key not in processed_queries:
                            clustered_results.append({
                                'query': item['query'],
                                'user': item['user'],
                                'topics': item['topics'],
                                'cluster': f"{topic}_-1",
                                'embedding': item['embedding']
                            })
                            processed_queries.add(query_key)
            
            # Assign noise queries
            for i, label in enumerate(labels):
                if label == -1:
                    item = group[i]
                    query_key = (item['query'], tuple(item['topics']))
                    if query_key not in processed_queries:
                        clustered_results.append({
                            'query': item['query'],
                            'user': item['user'],
                            'topics': item['topics'],
                            'cluster': f"{topic}_-1",
                            'embedding': item['embedding']
                        })
                        processed_queries.add(query_key)
                        
        except Exception as e:
            print(f"Error clustering topic '{topic}': {e}")
            # Assign all to noise
            for item in group:
                query_key = (item['query'], tuple(item['topics']))
                if query_key not in processed_queries:
                    clustered_results.append({
                        'query': item['query'],
                        'user': item['user'],
                        'topics': item['topics'],
                        'cluster': f"{topic}_-1",
                        'embedding': item['embedding']
                    })
                    processed_queries.add(query_key)
    
    # Create result DataFrame
    result_df = pd.DataFrame(clustered_results)
    
    # Remove embeddings to reduce file size
    if 'embedding' in result_df.columns:
        result_df = result_df.drop('embedding', axis=1)
    
    print(f"Clustering complete: {len(result_df)} queries assigned to clusters")
    print(f"Generated {len(cluster_metadata)} valid clusters")
    
    return result_df, cluster_metadata


def validate_cluster(
    cluster_items: List[Dict],
    cluster_embeddings: np.ndarray,
    topic: str,
    label: int,
    min_cluster_similarity: float,
    min_seo_similarity: float,
    min_unique_users: int,
    min_cluster_size: int
) -> Tuple[bool, Optional[Dict]]:
    """
    Validate a cluster and create metadata if valid.
    
    Args:
        cluster_items: List of query items in the cluster
        cluster_embeddings: Embeddings for the cluster
        topic: Topic name
        label: Cluster label
        min_cluster_similarity: Minimum similarity for valid clusters
        min_seo_similarity: Minimum similarity for SEO candidates
        min_unique_users: Minimum unique users per cluster
        min_cluster_size: Minimum queries per cluster
    
    Returns:
        Tuple of (is_valid, metadata)
    """
    n_queries = len(cluster_items)
    reasons = []
    if n_queries < min_cluster_size:
        reasons.append(f"too few queries: {n_queries} (min {min_cluster_size})")
    # Compute average pairwise cosine similarity
    sim_matrix = cosine_similarity(cluster_embeddings)
    avg_sim = (np.sum(sim_matrix) - n_queries) / (n_queries * (n_queries - 1))
    if avg_sim < min_cluster_similarity:
        reasons.append(f"low avg similarity: {avg_sim:.4f} (min {min_cluster_similarity})")
    # Check user diversity
    unique_users = set(item['user'] for item in cluster_items)
    if len(unique_users) < min_unique_users:
        reasons.append(f"not enough unique users: {len(unique_users)} (min {min_unique_users})")
    # Find most central query
    centroid = np.mean(cluster_embeddings, axis=0)
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    central_idx = int(np.argmin(distances))
    # Create metadata
    cluster_label = f"{topic}_{label}"
    seo_candidate = avg_sim >= min_seo_similarity
    metadata = {
        'topic': topic,
        'cluster_label': cluster_label,
        'n_queries': n_queries,
        'avg_similarity': round(float(avg_sim), 4),
        'unique_users': len(unique_users),
        'most_central_query': cluster_items[central_idx]['query'],
        'seo_candidate': str(seo_candidate).lower(),
        'rejection_reasons': reasons if reasons else None
    }
    if reasons:
        return False, metadata
    return True, metadata


def generate_cluster_report(clustered_df: pd.DataFrame, cluster_metadata: List[Dict]) -> str:
    """
    Generate a text report of clustering results.
    
    Args:
        clustered_df: DataFrame with cluster assignments
        cluster_metadata: List of cluster metadata dictionaries
    
    Returns:
        Formatted report string
    """
    from datetime import datetime
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("CLUSTER ANALYSIS REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # SEO candidates
    seo_candidates = [meta for meta in cluster_metadata if meta['seo_candidate'] == 'true']
    report_lines.append("=== SEO CANDIDATE CLUSTERS ===")
    for meta in seo_candidates:
        report_lines.append(f"\n📈 SEO Candidate: {meta['cluster_label']}")
        report_lines.append(f"   Topic: {meta['topic']}")
        report_lines.append(f"   Queries: {meta['n_queries']}")
        report_lines.append(f"   Avg Similarity: {meta['avg_similarity']}")
        report_lines.append(f"   Unique Users: {meta['unique_users']}")
        report_lines.append(f"   Representative: {meta['most_central_query']}")
    
    report_lines.append(f"\nTotal SEO Candidates: {len(seo_candidates)}")
    
    # All clusters
    report_lines.append("\n" + "=" * 60)
    report_lines.append("ALL CLUSTERS")
    report_lines.append("=" * 60)
    
    valid_clusters = [meta for meta in cluster_metadata if not meta['cluster_label'].endswith('_-1')]
    for meta in valid_clusters:
        report_lines.append(f"\n=== Cluster {meta['cluster_label']} ===")
        report_lines.append(f"Number of queries: {meta['n_queries']}")
        report_lines.append(f"Average similarity: {meta['avg_similarity']}")
        report_lines.append(f"Unique users: {meta['unique_users']}")
        report_lines.append(f"Representative: {meta['most_central_query']}")
    
    report_lines.append(f"\nTotal Valid Clusters: {len(valid_clusters)}")
    
    return '\n'.join(report_lines) 