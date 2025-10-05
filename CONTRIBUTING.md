# Contributing to torchflow

Thank you for your interest in contributing to torchflow! This document
describes the preferred workflow, testing, and coding standards for the
repository.

Getting started

1. Fork the repository: [https://github.com/mobadara/torchflow](https://github.com/mobadara/torchflow)
2. Create a feature branch:

```bash
git checkout -b feature/my-change
```

3. Implement your changes and add tests under `tests/`.

Coding guidelines

- Keep functions small and focused.
- Add docstrings for public functions and classes.
- Follow the existing style (PEP8). Use linters/formatters if available.

Testing

- Add unit tests for new behavior in `tests/`.
- Some tests skip when optional dependencies are missing (e.g., tensorboard,
  optuna). Use `pytest.importorskip('torch')` where appropriate.

Run tests locally:

```bash
pip install -e .[dev]
pytest -q
```

Pull requests

- Open a pull request against the `main` branch.
- Provide a clear description and link to any related issue.
- Keep changes small and focused when possible.

Code of conduct

Please follow standard community guidelines. Be respectful and constructive.

Thank you for contributing!
