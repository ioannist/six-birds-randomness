import numpy as np
import randomness_ledger


def test_imports_and_numpy() -> None:
    _ = randomness_ledger
    assert np.array([1]).sum() == 1
