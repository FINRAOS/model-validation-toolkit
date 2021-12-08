import os

from setuptools import find_packages, setup

VERSION = "0.0.1"

here = os.path.dirname(os.path.realpath(__file__))
extras_require = {
    "doc": [
        "nbsphinx",
        "sphinx",
        "sphinx-rtd-theme",
        "sphinxcontrib-bibtex",
        "imageio",
        "myst-parser",
    ]
}
setup(
    name="mvtk",
    version=VERSION,
    url="Apache License 2.0",
    author="Alex Eftimiades",
    author_email="alexeftimiades@gmail.com",
    description="Model validation toolkit",
    packages=find_packages(),
    install_requires=[
        "jax>=0.2.8",
        "public>=2020.12.3",
        "fastcore>=1.3.25",
        "jaxlib>=0.1.23",
        "scikit-learn",
        "numpy",
        "matplotlib",
        "scipy",
        "seaborn",
        "pandas>=0.23.4",
        "tqdm",
    ],
    extras_require=extras_require,
)
