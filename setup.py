from setuptools import find_packages, setup

_dct = {}
with open("mvtk/version.py") as f:
    exec(f.read(), _dct)
__version__ = _dct["__version__"]

extras_require = {
    "doc": [
        "nbsphinx",
        "sphinx",
        "sphinx-rtd-theme",
        "sphinxcontrib-bibtex",
        "imageio",
        "myst-parser",
    ],
    "pytorch": [
        "torch"
    ],
    "tensorflow": [
        "tensorflow"
    ]
}
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mvtk",
    version=__version__,
    license="Apache-2.0",
    author="Alex Eftimiades",
    author_email="alexeftimiades@gmail.com",
    description="Model validation toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
    ],
    install_requires=[
        "jax>=0.2.8,<=0.4.16",
        "public>=2020.12.3",
        "fastcore>=1.3.25",
        "jaxlib>=0.1.23,<=0.4.16",
        "scikit-learn",
        "numpy",
        "matplotlib",
        "scipy",
        "seaborn",
        "pandas>=0.23.4",
        "tqdm",
        "ray"
    ],
    extras_require=extras_require,
    url="https://finraos.github.io/model-validation-toolkit/",
    project_urls={
        "Bug Tracker": "https://github.com/FINRAOS/model-validation-toolkit/issues",
    },
)
