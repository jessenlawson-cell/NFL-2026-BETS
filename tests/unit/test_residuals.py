from __future__ import annotations

import numpy as np
import pytest

from nfl_bets.model.residuals import KEY_NUMBERS, EmpiricalResidualMapper


def test_spread_mapper_preserves_key_number_push_mass() -> None:
    actual = np.array([3.0] * 20 + [7.0] * 10 + list(np.linspace(-20, 20, 70)))
    projection = np.zeros_like(actual)
    mapper = EmpiricalResidualMapper(key_numbers=KEY_NUMBERS).fit(actual, projection)
    probabilities = mapper.probabilities(0.0, 3.0)
    assert probabilities.push >= 0.20
    assert probabilities.win + probabilities.push + probabilities.loss == pytest.approx(1.0)
