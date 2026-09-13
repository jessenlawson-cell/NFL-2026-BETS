from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

KEY_NUMBERS = (3, 6, 7, 10, 14)


@dataclass
class OutcomeProbabilities:
    win: float
    push: float
    loss: float


class EmpiricalResidualMapper:
    """Training-only empirical outcome mapper with explicit signed spread key mass."""

    def __init__(self, *, key_numbers: tuple[int, ...] = ()) -> None:
        self.key_numbers = key_numbers
        self.residuals: np.ndarray | None = None
        self.non_key_residuals: np.ndarray | None = None
        self.key_mass: dict[int, float] = {}

    def fit(self, actual: np.ndarray, projection: np.ndarray) -> EmpiricalResidualMapper:
        actual_array = np.asarray(actual, dtype=float)
        projection_array = np.asarray(projection, dtype=float)
        valid = np.isfinite(actual_array) & np.isfinite(projection_array)
        actual_array = actual_array[valid]
        projection_array = projection_array[valid]
        if actual_array.size < 30:
            raise ValueError("At least 30 training residuals are required")
        self.residuals = actual_array - projection_array
        signed_keys = {sign * key for key in self.key_numbers for sign in (-1, 1)}
        self.key_mass = {
            key: float(np.mean(np.isclose(actual_array, key)))
            for key in sorted(signed_keys)
            if np.any(np.isclose(actual_array, key))
        }
        non_key = (
            ~np.isin(actual_array, list(signed_keys))
            if signed_keys
            else np.ones(actual_array.size, bool)
        )
        self.non_key_residuals = self.residuals[non_key]
        if self.non_key_residuals.size == 0:
            self.non_key_residuals = self.residuals
            self.key_mass = {}
        return self

    def probabilities(self, projection: float, line: float) -> OutcomeProbabilities:
        if self.residuals is None or self.non_key_residuals is None:
            raise RuntimeError("Residual mapper is not fitted")
        simulated = projection + self.non_key_residuals
        key_weight = sum(self.key_mass.values())
        continuous_weight = max(0.0, 1.0 - key_weight)
        win = continuous_weight * float(np.mean(simulated > line))
        push = continuous_weight * float(np.mean(np.isclose(simulated, line)))
        loss = continuous_weight * float(np.mean(simulated < line))
        for key, mass in self.key_mass.items():
            if key > line:
                win += mass
            elif key < line:
                loss += mass
            else:
                push += mass
        total = win + push + loss
        return OutcomeProbabilities(win / total, push / total, loss / total)


class ProbabilityCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(C=1_000_000.0, solver="lbfgs")
        self.is_constant = False
        self.constant = 0.5

    def fit(self, probabilities: np.ndarray, outcomes: np.ndarray) -> ProbabilityCalibrator:
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        y = np.asarray(outcomes, dtype=int)
        if np.unique(y).size < 2:
            self.is_constant = True
            self.constant = float(np.mean(y))
            return self
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        self.model.fit(logits, y)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        if self.is_constant:
            return np.full_like(p, self.constant)
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        return np.asarray(self.model.predict_proba(logits)[:, 1], dtype=float)
