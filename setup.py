
from setuptools import find_packages, setup

install_requires = ["numpy",
                    "networkx",
                    "tqdm",
                    "python-louvain",
                    "scikit-learn",
                    "scipy",
                    "pygsp",
                    "gensim==3.8.3",
                    "pandas",
                    "six"]


setup_requires = ['pytest-runner']


tests_require = ['pytest',
                 'pytest-cov',
                 'mock']


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


setup(
  name="graph-tiger",
  packages=find_packages(),
  version="0.1.0",
  license="MIT",
  description="A general purpose library for graph vulnerability and robustness analysis.",
  author="Scott Freitas",
  author_email="safreita1@gmail.com",
  url="https://github.com/safreita1/TIGER",
  keywords=keywords,
  install_requires=install_requires,
  setup_requires=setup_requires,
  tests_require=tests_require,
  classifiers=["Development Status :: 3 - Alpha",
               "Intended Audience :: Developers",
               "License :: OSI Approved :: MIT License",
               "Programming Language :: Python :: 3.6"],
)