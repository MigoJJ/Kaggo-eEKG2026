import sys
import unittest
from pathlib import Path

import numpy as np


ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from ecgml.preprocess import znormalize_per_lead  # noqa: E402


class PreprocessContractTest(unittest.TestCase):
    def test_znormalize_matches_java_contract_fixture(self):
        samples = np.array(
            [
                [1, 2, 3, 4, 5],
                [2, 2, 2, 2, 2],
                [-1, 0, 1, 0, -1],
            ],
            dtype=np.float64,
        )
        expected = np.array(
            [
                [-1.41421356, -0.70710678, 0.0, 0.70710678, 1.41421356],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.06904497, 0.26726124, 1.60356745, 0.26726124, -1.06904497],
            ],
            dtype=np.float64,
        )

        actual = znormalize_per_lead(samples)

        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
