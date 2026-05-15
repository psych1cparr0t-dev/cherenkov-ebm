from setuptools import setup, find_packages

setup(
    name="cherenkov-ebm",
    version="4.3.0",
    author="Max Bradford",
    author_email="max@cherenkov.industries",
    description="Geometric reasoning layer via autonomous geometric primitive synthesis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/psych1cparr0t-dev/cherenkov-ebm",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "scipy>=1.10",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "matplotlib>=3.7"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    keywords="energy-based-model geometric-reasoning machine-learning parsing-layer",
)
