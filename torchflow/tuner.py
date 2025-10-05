"""
Hyperparameter tuning utilities using Optuna.

This module provides a tiny helper to run Optuna studies that build models
via a user-supplied ``build_fn(trial)`` and train them using ``torchflow.Trainer``. ``Optuna`` is imported lazily so importing this module
doesn't require Optuna to be installed.
"""
from typing import Callable, Any, Optional


def _ensure_optuna():
    try:
        import optuna

        return optuna
    except Exception as exc:  # pragma: no cover - runtime dependency handling
        raise RuntimeError(
            "optuna is required for tuning. Install it with `pip install optuna` or add it to your environment."
        ) from exc


def tune(
    build_fn: Callable[[Any], dict],
    train_loader,
    val_loader,
    num_epochs: int = 5,
    n_trials: int = 20,
    direction: str = "min",
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    n_jobs: int = 1,
    show_progress: bool = False,
):
    """
    Run an Optuna study over ``n_trials``.

    build_fn should accept an Optuna trial and return a dict with keys
    ``'model'``, ``'optimizer'``, ``'criterion'`` and optionally
    ``'device'``, ``'callbacks'``, ``'writer'``, ``'metrics'``,
    ``'mlflow_tracking'``.

    The objective returned to Optuna is the last validation loss recorded by
    the trainer (falls back to last training loss, or inf).
    """
    optuna = _ensure_optuna()

    study = optuna.create_study(
        direction=direction, study_name=study_name, storage=storage, load_if_exists=True
    )

    def objective(trial):
        cfg = build_fn(trial)

        if not all(k in cfg for k in ("model", "optimizer", "criterion")):
            raise ValueError("build_fn must return dict with keys: 'model','optimizer','criterion'")

        # lazy import Trainer to avoid import-time cycles
        from torchflow.trainer import Trainer

        model = cfg["model"]
        optimizer = cfg["optimizer"]
        criterion = cfg["criterion"]
        device = cfg.get("device", "cpu")
        callbacks = cfg.get("callbacks")
        writer = cfg.get("writer")
        metrics = cfg.get("metrics")
        mlflow_tracking = cfg.get("mlflow_tracking", False)

        trainer = Trainer(
            model,
            criterion,
            optimizer,
            device=device,
            metrics=metrics,
            writer=writer,
            mlflow_tracking=mlflow_tracking,
            callbacks=callbacks,
        )

        _, history, _ = trainer.train(train_loader, val_loader=val_loader, num_epochs=num_epochs)

        if "val_loss" in history and history["val_loss"]:
            return float(history["val_loss"][-1])
        if "train_loss" in history and history["train_loss"]:
            return float(history["train_loss"][-1])
        return float("inf")

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    if show_progress:
        best = study.best_trial
        print(f"Best trial: {best.number}, value={study.best_value}")
        print("Params:")
        for k, v in best.params.items():
            print(f"  {k}: {v}")

    return study


def example_build_fn(trial):
    """
    Tiny example build_fn for docs/tests.

    Samples a learning rate and a hidden-size, constructs a tiny MLP and
    returns the dict expected by ``tune``.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchflow.callbacks import EarlyStopping

    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    hidden = trial.suggest_int("hidden", 8, 64)

    model = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    early = EarlyStopping(monitor="val_loss", patience=2)

    return {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "device": "cpu",
        "callbacks": [early],
    }
