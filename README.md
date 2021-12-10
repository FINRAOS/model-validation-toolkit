<img src="docs/images/logo.svg" alt="drawing" width="250px"/>

[![CircleCI](https://circleci.com/gh/FINRAOS/model-validation-toolkit/tree/main.svg?style=svg)](https://circleci.com/gh/FINRAOS/Gatekeeper/tree/main)[![Join the chat at https://gitter.im/FINRAOS/model-validation-toolkit](https://badges.gitter.im/FINRAOS/model-validation-toolkit.svg)](https://gitter.im/FINRAOS/model-validation-toolkit?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)![PyPI version](https://img.shields.io/pypi/v/mvtk)

# Model Validation Tookit

## Installation

Run `pip install mvtk`.

**Windows users**: Until [Jaxlib is supported on windows
natively](https://github.com/google/jax/issues/438) you will need to either use
this library from you Linux subsystem or within a Docker container.
Alternatively, you can [build jaxlib from
source](https://jax.readthedocs.io/en/latest/developer.html#additional-notes-for-building-jaxlib-from-source-on-windows).

## Developers

Check out this repository and `cd` into the directory.

Run `pip install -e ".[doc]"`.

The `[doc]` is used to install dependencies for building documentation.

# Submodules
You can import:

- `mvtk.credibility` for assessing credibility from sample size.
- `mvtk.interprenet` for building interpretable neural nets.
- `mvtk.thresholding` for adaptive thresholding.
- `mvtk.sobol` for Sobol sensitivity analysis
- `mvtk.supervisor` for divergence anlysis
- `mvtk.metrics` for specialised metrics

# Documentation
You can run `make -C docs html` on a Mac or `make.bat -C docs html` on a PC to just rebuild the docs. In this case, point your browser to ```docs/_build/html/index.html``` to view the homepage. If your browser was already pointing to documentation that you changed, you can refresh the page to see the changes.
