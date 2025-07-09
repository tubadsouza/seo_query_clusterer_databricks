from setuptools import setup, find_packages

setup(
    name="seo_query_clusterer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "pandas",
        "numpy",
        "openai",
        "tqdm",
        # add any other dependencies your project uses
    ],
    python_requires=">=3.8",
)