[comment]: <> (Adapted from JAX's contribution guide)

# Contributing

Everyone can contribute to Model Validation Toolkit, and we value everyone's contributions. There are several
ways to contribute, including:


- Answering questions on Model Validation Toolkit's [gitter channel](https://gitter.im/FINRAOS/model-validation-toolkit)
- Improving or expanding Model Validation Toolkit's [documentation](https://finraos.github.io/model-validation-toolkit/docs/html/index.html)
- Contributing to Model Validation Toolkit's [code-base](https://github.com/FINRAOS/model-validation-toolkit/)

## Ways to contribute

We welcome pull requests, in particular for those issues marked with
[contributions welcome](https://github.com/FINRAOS/model-validation-toolkit/issues?q=is%3Aopen+is%3Aissue+label%3A%22contributions+welcome%22) or
[good first issue](https://github.com/FINRAOS/model-validation-toolkit/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).

For other proposals, we ask that you first open a GitHub
[Issue](https://github.com/FINRAOS/model-validation-toolkit/issues/new/choose) or
[Discussion](https://github.com/FINRAOS/model-validation-toolkit/discussions)
to seek feedback on your planned contribution.

## Contributing code using pull requests

We do all of our development using git, so basic knowledge is assumed.

Follow these steps to contribute code:

1. Fork the Model Validation Toolkit repository by clicking the **Fork** button on the
   [repository page](https://www.github.com/FINRAOS/model-validation-toolkit). This creates
   a copy of the Model Validation Toolkit repository in your own account.

2. Install Python >=3.6 locally in order to run tests.

3. `pip` installing your fork from source. This allows you to modify the code
   and immediately test it out:

   ```bash
   git clone https://github.com/YOUR_USERNAME/model-validation-toolkit
   cd model-validation-toolkit
   pip install -e .  # Installs Model Validation Toolkit from the current directory in editable mode.
   ```

4. Add the Model Validation Toolkit repo as an upstream remote, so you can use it to sync your
   changes.

   ```bash
   git remote add upstream http://www.github.com/FINRAOS/model-validation-toolkit
   ```

5. Create a branch where you will develop from:

   ```bash
   git checkout -b name-of-change
   ```

   And implement your changes using your favorite editor.

6. Make sure the tests pass by running the following command from the top of
   the repository:

   ```bash
   pytest tests/
   ```

   If you know the specific test file that covers your changes, you can limit the tests to that; for example:

   ```bash
   pytest tests/supervisor
   ```

   Model Validation Toolkit also offers more fine-grained control over which particular tests are run;
   see {ref}`running-tests` for more information.

7. Once you are satisfied with your change, create a commit as follows ([how to write a commit message](https://chris.beams.io/posts/git-commit/)):

   ```bash
   git add file1.py file2.py ...
   git commit -s -m "Your commit message"
   ```

   Please be sure to sign off your work when you commit it with the `-s` or, equivalently `--sign-off` flag to agree to our [DCO](https://raw.githubusercontent.com/FINRAOS/model-validation-toolkit/main/DCO).

   Then sync your code with the main repo:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Finally, push your commit on your development branch and create a remote 
   branch in your fork that you can use to create a pull request from:

   ```bash
   git push --set-upstream origin name-of-change
   ```

8. Create a pull request from the Model Validation Toolkit repository and send it for review.
   Check the {ref}`pr-checklist` for considerations when preparing your PR, and
   consult [GitHub Help](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)
   if you need more information on using pull requests.

## Model Validation Toolkit pull request checklist

As you prepare a Model Validation Toolkit pull request, here are a few things to keep in mind:

### DCO

By contributing to this project, you agree to our [DCO](https://raw.githubusercontent.com/FINRAOS/model-validation-toolkit/main/DCO).

### Single-change commits and pull requests

A git commit ought to be a self-contained, single change with a descriptive
message. This helps with review and with identifying or reverting changes if
issues are uncovered later on.

Pull requests typically comprise a single git commit. In preparing a pull
request for review, you may need to squash together multiple commits. We ask
that you do this prior to sending the PR for review if possible. The `git
rebase -i` command might be useful to this end.

### Linting and Type-checking

Model Validation Toolkit uses [mypy](https://mypy.readthedocs.io/) and [flake8](https://flake8.pycqa.org/)
to statically test code quality; the easiest way to run these checks locally is via
the [pre-commit](https://pre-commit.com/) framework:

```bash
pip install pre-commit
pre-commit run --all
```

### Full GitHub test suite

Your PR will automatically be run through a full test suite on GitHub CI, which
covers a range of Python versions, dependency versions, and configuration options.
It's normal for these tests to turn up failures that you didn't catch locally; to
fix the issues you can push new commits to your branch.
