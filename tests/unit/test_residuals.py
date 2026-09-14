from __future__ import annotations

import numpy as np
import pytest

from nfl_bets.model.residuals import (
    KEY_NUMBERS,
    EmpiricalResidualMapper,
    StratifiedResidualMapper,
)


def test_spread_mapper_preserves_key_number_push_mass() -> None:
    actual = np.array([3.0] * 20 + [7.0] * 10 + list(np.linspace(-20, 20, 70)))
    projection = np.zeros_like(actual)
    mapper = EmpiricalResidualMapper(key_numbers=KEY_NUMBERS).fit(actual, projection)
    probabilities = mapper.probabilities(0.0, 3.0)
    assert probabilities.push >= 0.20
    assert probabilities.win + probabilities.push + probabilities.loss == pytest.approx(1.0)


def test_stratum_probability_is_shrunk_toward_global_distribution() -> None:
    actual = np.concatenate([np.linspace(-10, 10, 300), np.full(60, 7.0)])
    projection = np.zeros_like(actual)
    strata = np.concatenate([np.zeros(300), np.full(60, 14.0)])
    mapper = StratifiedResidualMapper(
        key_numbers=KEY_NUMBERS, shrinkage=200.0, absolute_strata=True
    ).fit(actual, projection, strata)
    local = mapper.probabilities(0.0, 7.0, 14.0)
    global_only = mapper.global_mapper.probabilities(0.0, 7.0)
    assert local.push >= global_only.push
    assert local.win + local.push + local.loss == pytest.approx(1.0)
