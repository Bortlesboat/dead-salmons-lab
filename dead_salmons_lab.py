from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


def generate_dataset(
    n_samples: int = 300,
    input_dim: int = 16,
    signal_strength: float = 0.0,
    noise_scale: float = 1.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Generate a balanced binary dataset with an optional planted signal."""

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, input_dim))
    latent_score = signal_strength * x[:, 0] + noise_scale * rng.normal(size=n_samples)
    cutoff = float(np.median(latent_score))
    y = (latent_score >= cutoff).astype(np.int64)
    return {"X": x.astype(np.float64), "y": y, "latent_score": latent_score}


def random_mlp_features(
    x: np.ndarray,
    hidden_width: int = 128,
    depth: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Project inputs through a frozen random ReLU MLP."""

    rng = np.random.default_rng(seed)
    features = np.asarray(x, dtype=np.float64)

    for _ in range(depth):
        weights = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(max(features.shape[1], 1)),
            size=(features.shape[1], hidden_width),
        )
        bias = rng.normal(loc=0.0, scale=0.2, size=(hidden_width,))
        features = np.maximum(features @ weights + bias, 0.0)

    return _zscore_columns(features)


def summarize_feature_scan(
    features: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute per-feature correlation tests and multiple-testing summaries."""

    y_float = np.asarray(y, dtype=np.float64)
    feature_count = int(features.shape[1])
    correlations = np.zeros(feature_count, dtype=np.float64)
    p_values = np.ones(feature_count, dtype=np.float64)

    for idx in range(feature_count):
        if np.std(features[:, idx]) == 0:
            continue
        corr, p_value = pearsonr(features[:, idx], y_float)
        if not np.isfinite(corr):
            corr = 0.0
        if not np.isfinite(p_value):
            p_value = 1.0
        correlations[idx] = corr
        p_values[idx] = p_value

    _, bonferroni_adjusted, _, _ = multipletests(
        p_values,
        alpha=alpha,
        method="bonferroni",
    )
    fdr_mask, fdr_adjusted, _, _ = multipletests(
        p_values,
        alpha=alpha,
        method="fdr_bh",
    )

    return {
        "correlations": correlations,
        "p_values": p_values,
        "bonferroni_adjusted": bonferroni_adjusted,
        "fdr_adjusted": fdr_adjusted,
        "raw_significant": int(np.sum(p_values < alpha)),
        "bonferroni_significant": int(np.sum(bonferroni_adjusted < alpha)),
        "fdr_significant": int(np.sum(fdr_mask)),
        "min_p_value": float(np.min(p_values)),
        "top_feature_index": int(np.argmin(p_values)),
    }


def run_probe_hunt(
    x: np.ndarray,
    y: np.ndarray,
    hidden_width: int = 128,
    depth: int = 2,
    probe_trials: int = 16,
    permutation_trials: int = 24,
    test_size: float = 0.35,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate how impressive the best-picked probe looks under a fixed search budget."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    train_idx, test_idx = next(
        StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed,
        ).split(x, y)
    )
    feature_bank = [
        random_mlp_features(
            x,
            hidden_width=hidden_width,
            depth=depth,
            seed=seed + trial,
        )
        for trial in range(probe_trials)
    ]

    actual_accuracies: list[float] = []
    best_predictions: np.ndarray | None = None
    best_accuracy = -1.0

    for features in feature_bank:
        predictions = _fit_probe_and_predict(
            features[train_idx],
            y[train_idx],
            features[test_idx],
        )
        accuracy = float(np.mean(predictions == y[test_idx]))
        actual_accuracies.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_predictions = predictions

    permutation_rng = np.random.default_rng(seed + 10_000)
    permuted_best_accuracies: list[float] = []
    for _ in range(permutation_trials):
        y_perm = permutation_rng.permutation(y)
        best_perm_accuracy = 0.0
        for features in feature_bank:
            predictions = _fit_probe_and_predict(
                features[train_idx],
                y_perm[train_idx],
                features[test_idx],
            )
            accuracy = float(np.mean(predictions == y_perm[test_idx]))
            if accuracy > best_perm_accuracy:
                best_perm_accuracy = accuracy
        permuted_best_accuracies.append(best_perm_accuracy)

    if best_predictions is None:
        best_predictions = np.zeros_like(y[test_idx])

    accuracy_interval = _bootstrap_accuracy_interval(
        y[test_idx],
        best_predictions,
        seed=seed + 20_000,
    )
    permutation_best = np.asarray(permuted_best_accuracies, dtype=np.float64)
    actual = np.asarray(actual_accuracies, dtype=np.float64)
    permutation_p_value = (
        1.0 + float(np.sum(permutation_best >= best_accuracy))
    ) / (1.0 + float(len(permutation_best)))

    return {
        "actual_accuracies": actual,
        "best_accuracy": float(best_accuracy),
        "median_accuracy": float(np.median(actual)),
        "mean_accuracy": float(np.mean(actual)),
        "selection_gap": float(best_accuracy - np.median(actual)),
        "test_accuracy_interval": accuracy_interval,
        "permuted_best_accuracies": permutation_best,
        "null_95th_percentile": float(np.quantile(permutation_best, 0.95)),
        "permutation_p_value": float(permutation_p_value),
        "test_set_size": int(test_idx.size),
    }


def classify_regime(scan: dict[str, Any], probe: dict[str, Any]) -> str:
    """Classify the current run into null, weak-signal, or real-signal regimes."""

    low, _ = probe["test_accuracy_interval"]
    if probe["best_accuracy"] <= probe["null_95th_percentile"]:
        return "null"
    if low <= 0.5:
        return "weak_signal"
    return "real_signal"


def build_judge_summary(scan: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    """Return a compact summary object for judge-facing notebook copy."""

    regime = classify_regime(scan, probe)
    if regime == "null":
        headline = "Selection effect dressed up as structure."
    elif regime == "weak_signal":
        headline = "The effect clears the null, but it is still too fragile to trust."
    else:
        headline = "The guardrails are finally separating real signal from search artifacts."

    return {
        "regime": regime,
        "headline": headline,
        "summary_line": (
            f"Best probe {100 * probe['best_accuracy']:.1f}% vs search-matched null "
            f"95th percentile {100 * probe['null_95th_percentile']:.1f}%."
        ),
        "raw_hits": int(scan["raw_significant"]),
        "fdr_hits": int(scan["fdr_significant"]),
        "bonferroni_hits": int(scan["bonferroni_significant"]),
        "permutation_p_value": float(probe["permutation_p_value"]),
    }


def _fit_probe_and_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "probe",
                LogisticRegression(
                    max_iter=500,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model.predict(x_test)


def _bootstrap_accuracy_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    samples: int = 400,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    accuracies = np.empty(samples, dtype=np.float64)

    for idx in range(samples):
        sample_idx = rng.integers(0, len(y_true), size=len(y_true))
        accuracies[idx] = np.mean(y_true[sample_idx] == y_pred[sample_idx])

    low, high = np.quantile(accuracies, [0.025, 0.975])
    return float(low), float(high)


def _zscore_columns(features: np.ndarray) -> np.ndarray:
    centered = features - features.mean(axis=0, keepdims=True)
    scale = features.std(axis=0, keepdims=True)
    scale = np.where(scale == 0, 1.0, scale)
    return centered / scale
