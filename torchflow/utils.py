"""Small runtime utilities used by examples and tests.

This module is intentionally lightweight and avoids importing heavy
dependencies (like :mod:`torch` or :mod:`numpy`) at import time. Functions
that use those libraries perform lazy imports so the module can be imported
in environments where those extras are not installed.

Utilities included:
- set_seed(seed): set RNG seeds for reproducibility (random, numpy, torch when available)
- to_device(obj, device): move tensors / modules / nested containers to a device
- ensure_list(x): ensure the value is a list (wraps non-iterables)
"""
from typing import Any, Iterable, List, Optional
import random
import os


def set_seed(seed: Optional[int]) -> None:
	"""Set random seeds for reproducible runs.

	This sets the seed for Python's :mod:`random`, optionally NumPy and
	PyTorch if they are available. Passing ``None`` disables explicit seeding.

	Args:
		seed: integer seed or None to skip.
	"""
	if seed is None:
		return

	os.environ.setdefault("PYTHONHASHSEED", str(seed))
	random.seed(seed)

	try:
		import numpy as _np

		_np.random.seed(seed)
	except Exception:
		# numpy is optional
		pass

	try:
		import torch as _torch

		_torch.manual_seed(seed)
		if _torch.cuda.is_available():
			_torch.cuda.manual_seed_all(seed)
			# make deterministic where possible
			try:
				_torch.backends.cudnn.deterministic = True
				_torch.backends.cudnn.benchmark = False
			except Exception:
				pass
	except Exception:
		# torch is optional for this helper
		pass


def ensure_list(x: Any) -> List[Any]:
	"""Return ``x`` as a list. If ``x`` is ``None`` returns an empty list.

	Useful for normalizing API inputs that accept a single value or a list.
	"""
	if x is None:
		return []
	if isinstance(x, list):
		return x
	if isinstance(x, (tuple, set)):
		return list(x)
	return [x]


def _is_torch_tensor(obj: Any) -> bool:
	try:
		import torch

		return hasattr(torch, "is_tensor") and torch.is_tensor(obj)
	except Exception:
		return False


def to_device(obj: Any, device: Any):
	"""Move PyTorch tensors or modules (and nested containers) to ``device``.

	Works with tensors, ``torch.nn.Module``, lists, tuples and dicts. If
	PyTorch is not available this is a no-op.
	"""
	try:
		import torch

		# Module
		if isinstance(obj, torch.nn.Module):
			return obj.to(device)

		# Tensor
		if torch.is_tensor(obj):
			return obj.to(device)

		# Containers
		if isinstance(obj, dict):
			return {k: to_device(v, device) for k, v in obj.items()}
		if isinstance(obj, list):
			return [to_device(v, device) for v in obj]
		if isinstance(obj, tuple):
			return tuple(to_device(v, device) for v in obj)

		# Fallback: return as-is
		return obj
	except Exception:
		# torch not available or some other error — return original object
		return obj
