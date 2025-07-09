"""
SEO query generation functions using OpenAI GPT-4o-mini.
"""

import pandas as pd
from typing import List, Dict, Optional
import time
import os
from tqdm import tqdm


def generate_seo_queries(
    clustered_df: pd.DataFrame,
    queries_per_cluster: int = 1,
    api_key: Optional[str] = None,
    rate_limit_delay: float = 1.0
) -> pd.DataFrame:
    """
    Generate SEO queries for each cluster using GPT-4o-mini.
    
    Args:
        clustered_df: DataFrame with cluster assignments
        queries_per_cluster: Number of SEO queries to generate per cluster
        api_key: OpenAI API key (if None, uses environment variable)
        rate_limit_delay: Delay between API calls in seconds
    
    Returns:
        DataFrame with SEO queries for each cluster
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
    
    # Group queries by cluster
    clusters = {}
    for _, row in clustered_df.iterrows():
        cluster = row['cluster']
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(row['query'])
    
    # Filter out noise clusters
    valid_clusters = {k: v for k, v in clusters.items() if not k.endswith('_-1') and len(v) >= 2}
    
    print(f"Generating SEO queries for {len(valid_clusters)} clusters...")
    
    # Generate SEO queries
    seo_results = []
    for i, (cluster_label, queries) in enumerate(tqdm(valid_clusters.items(), desc="Generating SEO queries")):
        try:
            seo_queries = _generate_cluster_seo_queries(
                queries, cluster_label, queries_per_cluster
            )
            
            # Create rows for each SEO query
            for j, seo_query in enumerate(seo_queries):
                row = {
                    'cluster': cluster_label,
                    'seo_query': seo_query,
                    'query_number': j + 1,
                    'cluster_size': len(queries)
                }
                seo_results.append(row)
            
            # Rate limiting
            if i < len(valid_clusters) - 1:
                time.sleep(rate_limit_delay)
                
        except Exception as e:
            print(f"Error generating SEO queries for cluster {cluster_label}: {e}")
            # Add fallback query
            topic = cluster_label.split('_')[0] if '_' in cluster_label else cluster_label
            fallback_query = f"What is {topic}?"
            
            row = {
                'cluster': cluster_label,
                'seo_query': fallback_query,
                'query_number': 1,
                'cluster_size': len(queries)
            }
            seo_results.append(row)
    
    result_df = pd.DataFrame(seo_results)
    print(f"Generated {len(result_df)} SEO queries for {len(valid_clusters)} clusters")
    
    return result_df


def _generate_cluster_seo_queries(
    queries: List[str],
    cluster_label: str,
    queries_per_cluster: int
) -> List[str]:
    """
    Generate SEO queries for a single cluster using GPT-4o-mini.
    
    Args:
        queries: List of queries in the cluster
        cluster_label: Cluster identifier
        queries_per_cluster: Number of queries to generate
    
    Returns:
        List of generated SEO queries
    """
    import openai
    
    # Extract topic from cluster label
    topic = cluster_label.split('_')[0] if '_' in cluster_label else cluster_label
    
    # Prepare the prompt
    prompt = f"""
You are an SEO expert. Based on the following cluster of research queries, generate {queries_per_cluster} high-quality SEO search queries that would be typed into Google.

Cluster Topic: {topic}
Cluster Label: {cluster_label}

Research Queries in this cluster:
{chr(10).join([f"- {q}" for q in queries[:10]])}  # Limit to first 10 queries

Generate {queries_per_cluster} SEO search queries that:
1. Reflect common user search behavior (natural phrasing)
2. Capture core user intent (informational, problem-solving, comparative)
3. Represent the dominant theme in the cluster
4. Are realistic & specific (avoid vague phrasing)
5. Would be popular search terms for this topic

Format your response as exactly {queries_per_cluster} lines - just the SEO queries. No numbering, no explanations, just the queries.

Example format:
What are the symptoms of diabetes?
How to manage diabetes naturally?
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an SEO expert who creates natural, search-friendly queries based on research data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        # Extract queries from response
        generated_text = response.choices[0].message.content
        if generated_text:
            generated_text = generated_text.strip()
            # Take the first non-empty lines as queries
            lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
            if lines:
                return lines[:queries_per_cluster]
        
        # Fallback to generic queries
        if topic:
            return [f"What is {topic}?"] * queries_per_cluster
        else:
            return ["How to research this topic?"] * queries_per_cluster
        
    except Exception as e:
        print(f"Error in API call for cluster {cluster_label}: {e}")
        # Fallback to basic queries
        if topic:
            return [f"What is {topic}?"] * queries_per_cluster
        else:
            return ["How to research this topic?"] * queries_per_cluster


def analyze_seo_results(seo_df: pd.DataFrame) -> Dict:
    """
    Analyze SEO query generation results.
    
    Args:
        seo_df: DataFrame with SEO queries
    
    Returns:
        Dictionary with analysis statistics
    """
    if len(seo_df) == 0:
        return {'total_queries': 0, 'unique_clusters': 0, 'avg_cluster_size': 0}
    
    stats = {
        'total_queries': len(seo_df),
        'unique_clusters': seo_df['cluster'].nunique(),
        'avg_cluster_size': seo_df['cluster_size'].mean(),
        'queries_per_cluster': seo_df.groupby('cluster').size().mean(),
        'avg_query_length': seo_df['seo_query'].str.len().mean()
    }
    
    return stats 