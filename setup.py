import os
from setuptools import find_packages, setup

setup_requires = ["Cython", "pandas", "numpy", "pythran", "scikit-image"]

install_requires = ["joblib",
                    "tqdm",
                    "six",
                    "stopit",
                    "pandas",
                    "networkx",
                    "scipy",
                    "pillow",
                    "matplotlib",
                    "fa2",
                    "onnx",
                    "PyYAML",
                    "ipython",
                    "datashader",
                    "dask",
                    "bezier",
                    "ffmpeg"]


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
    version="0.1.6",
    license="MIT",
    description="A general purpose library for graph vulnerability and robustness analysis.",
    # long_description=long_description,
    # long_description_content_type='text/markdown',
    author="Scott Freitas",
    author_email="safreita1@gmail.com",
    url="https://github.com/safreita1/TIGER",
    download_url="https://github.com/safreita1/TIGER/archive/0.1.6.tar.gz",
    keywords=keywords,
    setup_requires=setup_requires,
    install_requires=install_requires,
    classifiers=["Development Status :: 3 - Alpha",
                 "Intended Audience :: Developers",
                 "License :: OSI Approved :: MIT License",
                 "Programming Language :: Python :: 3.6"],
)
