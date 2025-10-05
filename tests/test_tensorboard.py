import pytest
pytest.importorskip('torch')

import sys
sys.path.insert(0, '/home/mobadara/Documents/projects/torchflow')

from torchflow.callbacks import make_summary_writer


def test_make_summary_writer_safety():
    # The factory should not raise even if tensorboard/torch are missing.
    w = make_summary_writer(None)
    # It's valid for environments without tensorboard to return None
    assert (w is None) or hasattr(w, 'add_scalar')
