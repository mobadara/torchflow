# torchflow

A lightweight, dependency-minimal PyTorch training framework that provides a
clean Trainer API, a set of commonly-used training callbacks, and an
Optuna-backed tuner helper for hyperparameter search.

This project (github.com/mobadara/torchflow) is designed to be published on PyPI
and used as a small building block for training experiments and demos where
# torchflow

A lightweight, dependency-minimal PyTorch training framework that provides a
clean Trainer API, a set of commonly-used training callbacks, and an
Optuna-backed tuner helper for hyperparameter search.

This project ([github.com/mobadara/torchflow](https://github.com/mobadara/torchflow))
is designed to be published on PyPI and used as a small building block for
training experiments and demos where you want sensible defaults and pluggable
callbacks without the overhead of a large framework.

## Contents

- Features
- Installation
- Quick start
- Callbacks
- Tuner (Optuna)
- Examples
- Testing
- Contributing
- License

## Features

- Simple, readable `Trainer` for training and validation loops.
- Callback system with lifecycle hooks (on_train_begin, on_epoch_begin, on_batch_end,
	on_validation_end, on_epoch_end, on_train_end).
- Built-in callbacks: EarlyStopping, ModelCheckpoint, LearningRateScheduler,
	ReduceLROnPlateau, CSVLogger, TensorBoardCallback, and more.
- Safe, lazy imports for optional heavy dependencies (TensorBoard, Optuna) so
	importing the library doesn't require installing every optional package.
- Small Optuna `tuner` helper that builds a new Trainer for each trial using
	a user-supplied `build_fn(trial)`.

## Installation

Install the core package from PyPI (when released):

```bash
pip install torchflow
```

For development from source:

```bash
git clone https://github.com/mobadara/torchflow.git
cd torchflow
pip install -e .
```

Optional extras:

- TensorBoard logging: `pip install tensorboard`
- Hyperparameter tuning: `pip install optuna`

## Quick start

Minimal training example (pseudo-code):

```python
import torch
from torch import nn, optim
from torchflow.trainer import Trainer

model = nn.Sequential(nn.Linear(10, 1))
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)

trainer = Trainer(model, criterion, optimizer, device='cpu')
trainer.train(train_loader, val_loader=val_loader, num_epochs=5)
```

## Callbacks

Callbacks are simple objects with lifecycle hooks that the `Trainer` calls at
key moments during training. They are passed to `Trainer` as a list and can
perform logging, checkpointing, learning-rate changes, early stopping, and
more.

Example with TensorBoard logging and early stopping:

```python
from torchflow.callbacks import TensorBoardCallback, EarlyStopping

tb = TensorBoardCallback(log_dir='runs/myrun')  # uses a safe SummaryWriter factory
early = EarlyStopping(monitor='val_loss', patience=3)

trainer = Trainer(model, criterion, optimizer, callbacks=[tb, early])
trainer.train(train_loader, val_loader=val_loader, num_epochs=20)
```

The library exposes a few convenience callbacks out of the box:

- EarlyStopping
- ModelCheckpoint
- LearningRateScheduler
- ReduceLROnPlateau
- CSVLogger
- TensorBoardCallback

## Tuner (Optuna)

`torchflow.tuner` provides a small wrapper around Optuna. The contract is:

- `build_fn(trial)` should return a dict with at least `model`, `optimizer`, and `criterion`.
- Optional keys `device`, `callbacks`, `writer`, `metrics`, `mlflow_tracking` may also be returned.

Example usage:

```python
from torchflow.tuner import tune, example_build_fn

# `example_build_fn` is a tiny helper included for demonstration.
study = tune(example_build_fn, train_loader, val_loader, n_trials=10, num_epochs=3)
```

The tuner imports Optuna lazily; importing `torchflow.tuner` does not require
Optuna to be installed. Calling `tune()` will raise a clear error if Optuna is
missing.

## Examples

Run the included example scripts in the `examples/` directory:

```bash
python examples/simple_train.py
python examples/lr_and_logging.py
python examples/tensorboard_example.py
```

Note: `examples/tensorboard_example.py` will try to use TensorBoard; install
the `tensorboard` package if you want to run it.

## Testing

Tests use `pytest` and are located in the `tests/` directory. Some tests skip
when optional dependencies (like `torch` or `tensorboard`) are not available.

Run tests locally:

```bash
pip install -e .[dev]
pytest -q
```

## Contributing

Contributions are welcome. See `CONTRIBUTING.md` for contribution
guidelines, the project's coding conventions, and testing instructions.

## License

This project is released under the terms of the license in the `LICENSE`
file. By contributing you agree to license your changes under the same terms.

## Maintainers

- Author: mobadara (@m_obadara)
- Repository: [https://github.com/mobadara/torchflow](https://github.com/mobadara/torchflow)

If you'd like to contact the maintainer, open an issue or mention the handle
on Twitter: [@m_obadara](https://twitter.com/m_obadara)

## Project & Contact

- GitHub: [https://github.com/mobadara/torchflow](https://github.com/mobadara/torchflow)
- Author: mobadara
- Twitter: [https://twitter.com/m_obadara](https://twitter.com/m_obadara)

---

If you'd like, I can also:
- Add a short `setup.cfg` / `pyproject.toml` example for PyPI metadata.
- Add a minimal `CHANGELOG.md` and `CONTRIBUTING.md`.
trainer = Trainer(model, criterion, optimizer, callbacks=[tb, early])
trainer.train(train_loader, val_loader=val_loader, num_epochs=20)
```

The library exposes a few convenience callbacks out of the box:

- EarlyStopping
- ModelCheckpoint
- LearningRateScheduler
- ReduceLROnPlateau
- CSVLogger
- TensorBoardCallback

## Tuner (Optuna)

`torchflow.tuner` provides a small wrapper around Optuna. The contract is:

- `build_fn(trial)` should return a dict with at least `model`, `optimizer`, and `criterion`.
- Optional keys `device`, `callbacks`, `writer`, `metrics`, `mlflow_tracking` may also be returned.

Example usage:

```python
from torchflow.tuner import tune, example_build_fn

# `example_build_fn` is a tiny helper included for demonstration.
study = tune(example_build_fn, train_loader, val_loader, n_trials=10, num_epochs=3)
```

The tuner imports Optuna lazily; importing `torchflow.tuner` does not require
Optuna to be installed. Calling `tune()` will raise a clear error if Optuna is
missing.

## Examples

Run the included example scripts in the `examples/` directory:

```bash
python examples/simple_train.py
python examples/lr_and_logging.py
python examples/tensorboard_example.py
```

Note: `examples/tensorboard_example.py` will try to use TensorBoard; install
the `tensorboard` package if you want to run it.

## Testing

Tests use `pytest` and are located in the `tests/` directory. Some tests skip
when optional dependencies (like `torch` or `tensorboard`) are not available.

Run tests locally:

```bash
pip install -e .[dev]
pytest -q
```

## Contributing

Contributions are welcome. See `CONTRIBUTING.md` for contribution
guidelines, the project's coding conventions, and testing instructions.

## License

This project is released under the terms of the license in the `LICENSE`
file. By contributing you agree to license your changes under the same terms.

## Maintainers

- Author: mobadara (@m_obadara)
- Repository: [https://github.com/mobadara/torchflow](https://github.com/mobadara/torchflow)

If you'd like to contact the maintainer, open an issue or mention the handle
on Twitter: [@m_obadara](https://twitter.com/m_obadara)

## Project & Contact

- GitHub: [https://github.com/mobadara/torchflow](https://github.com/mobadara/torchflow)
- Author: mobadara
- Twitter: [https://twitter.com/m_obadara](https://twitter.com/m_obadara)

---

If you'd like, I can also:
- Add a short `setup.cfg` / `pyproject.toml` example for PyPI metadata.
- Add a minimal `CHANGELOG.md` and `CONTRIBUTING.md`.

