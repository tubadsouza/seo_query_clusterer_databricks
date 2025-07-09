# Query Analyzer - Modular Pipeline

A modular, Databricks-compatible Python pipeline for processing, clustering, and analyzing user queries to identify SEO opportunities and search intent patterns.

## 🎯 Features

- **Modular Design**: Clean, importable functions with clear signatures
- **Databricks Compatible**: DBFS file operations and notebook integration
- **Flexible Input**: Accept DataFrames or file paths
- **Configurable**: Easy parameter tuning for different use cases
- **Production Ready**: Error handling, progress tracking, and logging
- **Memory Efficient**: Embedding removal to reduce file sizes

## 📦 Installation

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/query-analyzer.git
cd query-analyzer

# Install the package
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### Databricks Installation

```python
# In a Databricks notebook
!pip install query-analyzer
```

## 🚀 Quick Start

### Basic Usage

```python
from query_analyzer import run_pipeline

# Run complete pipeline
results = run_pipeline(
    input_path="/dbfs/path/to/queries.csv",
    output_dir="/dbfs/query_analyzer/output"
)

# Access results
print(f"Processed {results['statistics']['total_queries']} queries")
print(f"Created {results['statistics']['valid_clusters']} clusters")
```

### Notebook Integration

```python
import pandas as pd
from query_analyzer import run_pipeline_from_notebook

# Load your data
input_df = spark.read.csv("/dbfs/data/queries.csv", header=True).toPandas()

# Run pipeline
results = run_pipeline_from_notebook(
    input_df=input_df,
    output_dir="/dbfs/output",
    queries_per_cluster=2
)
```

## 📋 API Reference

### Core Functions

#### `run_pipeline()`
Main orchestration function that runs the complete pipeline.

```python
def run_pipeline(
    input_path: Optional[str] = None,
    input_df: Optional[pd.DataFrame] = None,
    output_dir: str = "/dbfs/query_analyzer/output",
    save_intermediate: bool = True,
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
) -> Dict[str, Any]
```

#### `embed_queries()`
Generate embeddings for a list of queries.

```python
def embed_queries(
    queries: List[str],
    batch_size: int = 100,
    model: str = "text-embedding-3-large",
    api_key: Optional[str] = None,
    rate_limit_delay: float = 1.0
) -> np.ndarray
```

#### `cluster_queries()`
Cluster queries using HDBSCAN with topic-based grouping.

```python
def cluster_queries(
    embeddings: np.ndarray,
    queries_df: pd.DataFrame,
    min_cluster_size: int = 3,
    max_cluster_size: int = 5,
    min_cluster_similarity: float = 0.6,
    min_seo_similarity: float = 0.8,
    min_unique_users: int = 3,
    cluster_selection_epsilon: float = 0.1
) -> Tuple[pd.DataFrame, List[Dict]]
```

#### `generate_seo_queries()`
Generate SEO queries for each cluster using GPT-4o-mini.

```python
def generate_seo_queries(
    clustered_df: pd.DataFrame,
    queries_per_cluster: int = 1,
    api_key: Optional[str] = None,
    rate_limit_delay: float = 1.0
) -> pd.DataFrame
```

### Utility Functions

#### `load_and_filter_queries()`
Load and filter queries from CSV or DataFrame.

```python
def load_and_filter_queries(
    file_path: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    min_length: int = 30,
    require_topics: bool = True,
    language_filter: bool = True
) -> pd.DataFrame
```

#### DBFS File Operations

```python
from query_analyzer.utils import read_csv_dbfs, write_csv_dbfs

# Read CSV with DBFS support
df = read_csv_dbfs("/dbfs/path/to/file.csv")

# Write CSV with DBFS support
write_csv_dbfs(df, "/dbfs/output/result.csv")
```

## 🔧 Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your-openai-api-key

# Optional
QUERY_ANALYZER_OUTPUT_DIR=/dbfs/query_analyzer/output
```

### Pipeline Parameters

```python
# Default configuration
config = {
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
```

## 📊 Input/Output Formats

### Input CSV Format

```csv
query,user_id,topic
"how does diabetes affect the body?","user123","Medicine,Health"
"what are diabetes symptoms?","user456","Medicine,Health"
```

### Output Files

The pipeline generates several output files:

- `filtered_queries.csv`: Cleaned and filtered queries
- `clustered_queries.csv`: Queries with cluster assignments
- `cluster_metadata.json`: Cluster statistics and metadata
- `cluster_report.txt`: Human-readable cluster analysis
- `seo_queries.csv`: Generated SEO queries

### Output DataFrame Structure

```python
# Clustered queries
clustered_df = pd.DataFrame({
    'query': ['how does diabetes affect the body?'],
    'user': ['user123'],
    'topics': [['Medicine', 'Health']],
    'cluster': ['Medicine_0']
})

# SEO queries
seo_df = pd.DataFrame({
    'cluster': ['Medicine_0'],
    'seo_query': ['What are the symptoms of diabetes?'],
    'query_number': [1],
    'cluster_size': [5]
})
```

## 🏗️ Architecture

### Module Structure

```
query_analyzer/
├── __init__.py          # Package exports
├── pipeline.py          # Main orchestration
├── data_loading.py      # Query filtering and preprocessing
├── embedding.py         # OpenAI embedding generation
├── clustering.py        # HDBSCAN clustering
├── seo_generation.py    # GPT-4o-mini SEO query generation
└── utils.py            # DBFS utilities and helpers
```

### Pipeline Flow

1. **Data Loading**: Filter and clean queries
2. **Embedding**: Generate OpenAI embeddings
3. **Clustering**: Group similar queries by topic
4. **SEO Generation**: Create search-optimized queries

## 🚀 Databricks Integration

### Notebook Example

```python
# MAGIC %md
# MAGIC # Query Analyzer Pipeline

# COMMAND ----------

from query_analyzer import run_pipeline_from_notebook

# Load data
input_df = spark.read.csv("/dbfs/data/queries.csv", header=True).toPandas()

# Run pipeline
results = run_pipeline_from_notebook(
    input_df=input_df,
    output_dir="/dbfs/output",
    queries_per_cluster=2
)

# Display results
display(results['seo_queries'])
```

### Scheduled Jobs

1. Create a Databricks job
2. Add your notebook as a task
3. Set schedule (daily, weekly, etc.)
4. Configure cluster settings
5. Monitor job runs

## 🔍 Advanced Usage

### Custom Embedding Models

```python
embeddings = embed_queries(
    queries=queries_list,
    model="text-embedding-ada-002",  # Alternative model
    batch_size=50
)
```

### Custom Clustering Parameters

```python
clustered_df, metadata = cluster_queries(
    embeddings=embeddings,
    queries_df=filtered_df,
    min_cluster_size=2,
    max_cluster_size=10,
    min_cluster_similarity=0.5,
    min_seo_similarity=0.7
)
```

### Multiple SEO Queries per Cluster

```python
seo_df = generate_seo_queries(
    clustered_df=clustered_df,
    queries_per_cluster=3  # Generate 3 queries per cluster
)
```

## 🛠️ Development

### Local Development

```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Format code
black query_analyzer/

# Lint code
flake8 query_analyzer/
```

### Adding New Features

1. Create new module in `query_analyzer/`
2. Add function to `__init__.py`
3. Update pipeline if needed
4. Add tests
5. Update documentation

## 📈 Performance

### Optimization Tips

- **Batch Size**: Increase `batch_size` for faster embedding generation
- **Rate Limiting**: Adjust `rate_limit_delay` based on API limits
- **Cluster Size**: Tune `min_cluster_size` and `max_cluster_size`
- **Similarity Thresholds**: Adjust based on your data quality

### Memory Management

- Embeddings are automatically removed from output files
- Use `save_intermediate=False` to reduce disk usage
- Process large datasets in chunks if needed

## 🔒 Security

- API keys are read from environment variables
- No hardcoded credentials in the code
- DBFS paths are validated before use
- Error handling prevents data leakage

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For questions or issues:
- Create an issue on GitHub
- Check the documentation
- Review the examples 