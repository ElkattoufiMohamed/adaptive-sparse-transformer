from setuptools import find_packages, setup

setup(
    name="adaptive-sparse-transformer",
    version="0.1.0",
    description="Novel Adaptive Sparse Attention Transformer Architecture",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Mohamed Elkattoufi",
    author_email="contact@mohamedelkattoufi.com",
    url="https://github.com/ElkattoufiMohamed/adaptive-sparse-transformer.git",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "wandb>=0.15.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
