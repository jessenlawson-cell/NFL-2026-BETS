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


class StratifiedResidualMapper:
    """Tercile residual mapper shrunk toward a global empirical distribution."""

    def __init__(
        self,
        *,
        key_numbers: tuple[int, ...] = (),
        shrinkage: float = 200.0,
        absolute_strata: bool = False,
    ) -> None:
        self.key_numbers = key_numbers
        self.shrinkage = shrinkage
        self.absolute_strata = absolute_strata
        self.edges: np.ndarray | None = None
        self.global_mapper = EmpiricalResidualMapper(key_numbers=key_numbers)
        self.strata: dict[int, tuple[EmpiricalResidualMapper, int]] = {}

    def fit(
        self, actual: np.ndarray, projection: np.ndarray, stratum_value: np.ndarray
    ) -> StratifiedResidualMapper:
        actual_array = np.asarray(actual, dtype=float)
        projection_array = np.asarray(projection, dtype=float)
        values = np.asarray(stratum_value, dtype=float)
        if self.absolute_strata:
            values = np.abs(values)
        valid = np.isfinite(actual_array) & np.isfinite(projection_array) & np.isfinite(values)
        actual_array = actual_array[valid]
        projection_array = projection_array[valid]
        values = values[valid]
        self.global_mapper.fit(actual_array, projection_array)
        self.edges = np.unique(np.quantile(values, [1.0 / 3.0, 2.0 / 3.0]))
        bins = np.digitize(values, self.edges, right=True)
        self.strata = {}
        for index in np.unique(bins):
            mask = bins == index
            count = int(mask.sum())
            if count < 30:
                continue
            mapper = EmpiricalResidualMapper(key_numbers=self.key_numbers).fit(
                actual_array[mask], projection_array[mask]
            )
            self.strata[int(index)] = (mapper, count)
        return self

    def probabilities(
        self, projection: float, line: float, stratum_value: float | None = None
    ) -> OutcomeProbabilities:
        if self.edges is None:
            raise RuntimeError("Stratified residual mapper is not fitted")
        value = line if stratum_value is None else stratum_value
        if self.absolute_strata:
            value = abs(value)
        index = int(np.digitize([value], self.edges, right=True)[0])
        global_probability = self.global_mapper.probabilities(projection, line)
        local = self.strata.get(index)
        if local is None:
            return global_probability
        mapper, count = local
        local_probability = mapper.probabilities(projection, line)
        weight = count / (count + self.shrinkage)
        return OutcomeProbabilities(
            win=weight * local_probability.win + (1.0 - weight) * global_probability.win,
            push=weight * local_probability.push + (1.0 - weight) * global_probability.push,
            loss=weight * local_probability.loss + (1.0 - weight) * global_probability.loss,
        )


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
