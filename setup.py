from setuptools import setup, find_packages

setup(
    name="a2a",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "fastapi",
        "uvicorn",
        "sentence-transformers",
        "faiss-cpu",
        "psycopg",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Agent-to-Agent matching and communication library",
    python_requires=">=3.8",
)