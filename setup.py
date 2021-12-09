from setuptools import find_packages, setup

VERSION = "0.0.1"

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
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mvtk",
    version=VERSION,
    license='Apache-2.0',
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
    url="https://finraos.github.io/model-validation-toolkit/",
    project_urls={
        "Bug Tracker": "https://github.com/FINRAOS/model-validation-toolkit/issues",
    },
)
