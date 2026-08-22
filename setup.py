import os
from setuptools import find_packages, setup

# run "git tag <version>" and then "git push origin master <version> when releasing a package to PyPi
version = "0.4.0"

keywords = ["data-science",
            "machine-learning",
            "networkx",
            "graph",
            "data-mining",
            "attack",
            "simulation",
            "vulnerability",
            "networks",
            "epidemics",
            "defense",
            "graph-mining",
            "diffusion",
            "robustness",
            "graph-attack",
            "adversarial-attacks",
            "network-attack",
            "cascading-failures",
            "netshield"]

cwd = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(cwd, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="graph-tiger",
    packages=find_packages(),
    version=version,
    license="MIT",
    description="A general purpose library for graph vulnerability and robustness analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Scott Freitas",
    author_email="safreita1@gmail.com",
    url="https://github.com/safreita1/TIGER",
    download_url="https://github.com/safreita1/TIGER/archive/{}.tar.gz".format(version),
    keywords=keywords,
    python_requires=">=3.8",
    install_requires=[
        "matplotlib>=3.4",
        "networkx>=2.6",
        "numpy>=1.20",
        "pandas>=1.3",
        "scipy>=1.7",
        "stopit>=1.1.2"
    ],
    extras_require={
        "test": ["pytest>=7", "pytest-cov>=4"],
        "visualization": ["dask>=2022", "datashader>=0.13", "fa2", "pillow>=8", "scikit-image>=0.19"]
    },
    classifiers=["Development Status :: 3 - Alpha",
                 "Intended Audience :: Developers",
                 "License :: OSI Approved :: MIT License",
                 "Programming Language :: Python :: 3",
                 "Programming Language :: Python :: 3.8",
                 "Programming Language :: Python :: 3.9",
                 "Programming Language :: Python :: 3.10",
                 "Programming Language :: Python :: 3.11",
                 "Programming Language :: Python :: 3.12",
                 "Programming Language :: Python :: 3.13",
                 "Programming Language :: Python :: 3.14"]
)
