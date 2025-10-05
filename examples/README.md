# Examples

This folder contains small runnable examples demonstrating how to use the `Trainer` and callbacks shipped with torchflow.

Files:

- `simple_train.py` — minimal training loop using `Trainer` with a `LambdaCallback` and `EarlyStopping`.
- `lr_and_logging.py` — shows `LearningRateScheduler` (wrapping a PyTorch scheduler), `ReduceLROnPlateau`, and `CSVLogger`.

Run an example:

```bash
python examples/simple_train.py
python examples/lr_and_logging.py
```

Notes:
- If you want TensorBoard logging, install `tensorboard` and use `TensorBoardCallback(log_dir='runs/myrun')` when creating the `Trainer`.
- CI uses pytest: `.github/workflows/ci.yml` runs tests on pushes/PRs to `main`/`master`.
