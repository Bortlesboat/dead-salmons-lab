# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "marimo==0.22.5",
#   "matplotlib==3.10.8",
#   "numpy==2.1.1",
#   "scikit-learn==1.5.2",
#   "scipy==1.15.3",
#   "statsmodels==0.14.6",
# ]
# ///

import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pearsonr
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from statsmodels.stats.multitest import multipletests

    plt.style.use("seaborn-v0_8-whitegrid")

    try:
        from dead_salmons_lab import (
            build_judge_summary,
            classify_regime,
            generate_dataset,
            random_mlp_features,
            run_probe_hunt,
            summarize_feature_scan,
        )
    except ModuleNotFoundError:
        def generate_dataset(
            n_samples: int = 300,
            input_dim: int = 16,
            signal_strength: float = 0.0,
            noise_scale: float = 1.0,
            seed: int = 0,
        ) -> dict[str, np.ndarray]:
            rng = np.random.default_rng(seed)
            x = rng.normal(size=(n_samples, input_dim))
            latent_score = signal_strength * x[:, 0] + noise_scale * rng.normal(size=n_samples)
            cutoff = float(np.median(latent_score))
            y = (latent_score >= cutoff).astype(np.int64)
            return {"X": x.astype(np.float64), "y": y, "latent_score": latent_score}

        def _zscore_columns(features: np.ndarray) -> np.ndarray:
            centered = features - features.mean(axis=0, keepdims=True)
            scale = features.std(axis=0, keepdims=True)
            scale = np.where(scale == 0, 1.0, scale)
            return centered / scale

        def random_mlp_features(
            x: np.ndarray,
            hidden_width: int = 128,
            depth: int = 2,
            seed: int = 0,
        ) -> np.ndarray:
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
        ) -> dict[str, object]:
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

        def _fit_probe_and_predict(
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
        ) -> np.ndarray:
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("probe", LogisticRegression(max_iter=500, solver="lbfgs")),
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

        def run_probe_hunt(
            x: np.ndarray,
            y: np.ndarray,
            hidden_width: int = 128,
            depth: int = 2,
            probe_trials: int = 16,
            permutation_trials: int = 24,
            test_size: float = 0.35,
            seed: int = 0,
        ) -> dict[str, object]:
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

        def classify_regime(scan: dict[str, object], probe: dict[str, object]) -> str:
            low, _ = probe["test_accuracy_interval"]
            if probe["best_accuracy"] <= probe["null_95th_percentile"]:
                return "null"
            if low <= 0.5:
                return "weak_signal"
            return "real_signal"

        def build_judge_summary(scan: dict[str, object], probe: dict[str, object]) -> dict[str, object]:
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


@app.cell
def _(mo, np, plt):
    def pct(value: float) -> str:
        return f"{100 * value:.1f}%"

    def scenario_name(signal_strength: float) -> str:
        if signal_strength <= 0.05:
            return "Null regime: every apparent discovery is false by construction."
        if signal_strength < 0.9:
            return "Weak-signal regime: there is some structure, but selection effects still distort the story."
        return "Real-signal regime: the same guardrails should finally start earning trust."

    def build_feature_figure(scan: dict[str, object]):
        correlations = np.asarray(scan["correlations"])
        p_values = np.asarray(scan["p_values"])
        raw_mask = p_values < 0.05
        highlight_index = int(scan["top_feature_index"])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

        axes[0].hist(
            p_values,
            bins=np.linspace(0, 1, 21),
            color="#4C78A8",
            edgecolor="white",
            alpha=0.9,
        )
        axes[0].axvline(0.05, color="#E45756", linestyle="--", linewidth=2)
        axes[0].set_title("Feature-level p-values")
        axes[0].set_xlabel("p-value")
        axes[0].set_ylabel("Hidden units")
        axes[0].text(
            0.05,
            axes[0].get_ylim()[1] * 0.88,
            "raw threshold",
            color="#E45756",
            fontsize=10,
        )

        axes[1].scatter(
            np.arange(len(correlations)),
            correlations,
            c=np.where(raw_mask, "#E45756", "#4C78A8"),
            alpha=0.75,
            s=24,
        )
        axes[1].scatter(
            [highlight_index],
            [correlations[highlight_index]],
            s=110,
            color="#F58518",
            edgecolors="black",
            linewidth=0.8,
            label="lowest p-value",
        )
        axes[1].axhline(0.0, color="#888888", linewidth=1)
        axes[1].set_title("Correlations across frozen random features")
        axes[1].set_xlabel("Hidden-unit index")
        axes[1].set_ylabel("Pearson r with label")
        axes[1].legend(loc="upper right")

        fig.tight_layout()
        return fig

    def build_probe_figure(probe: dict[str, object]):
        actual = np.asarray(probe["actual_accuracies"])
        null_best = np.asarray(probe["permuted_best_accuracies"])
        best_accuracy = float(probe["best_accuracy"])
        null_cutoff = float(probe["null_95th_percentile"])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

        axes[0].hist(
            actual,
            bins=np.linspace(0.35, 0.95, 15),
            color="#4C78A8",
            edgecolor="white",
            alpha=0.9,
        )
        axes[0].axvline(0.5, color="#888888", linestyle=":", linewidth=2)
        axes[0].axvline(best_accuracy, color="#E45756", linestyle="--", linewidth=2)
        axes[0].set_title("All searched probe accuracies")
        axes[0].set_xlabel("Held-out accuracy")
        axes[0].set_ylabel("Probe count")

        axes[1].hist(
            null_best,
            bins=np.linspace(0.35, 0.95, 15),
            color="#72B7B2",
            edgecolor="white",
            alpha=0.9,
        )
        axes[1].axvline(best_accuracy, color="#E45756", linestyle="--", linewidth=2)
        axes[1].axvline(null_cutoff, color="#54A24B", linestyle=":", linewidth=2)
        axes[1].set_title("Search-matched null: best result after label shuffling")
        axes[1].set_xlabel("Best null accuracy")
        axes[1].set_ylabel("Permutation runs")

        fig.tight_layout()
        return fig

    def top_feature_rows(scan: dict[str, object]) -> list[dict[str, object]]:
        order = np.argsort(np.asarray(scan["p_values"]))[:8]
        rows: list[dict[str, object]] = []
        for rank, index in enumerate(order, start=1):
            rows.append(
                {
                    "rank": rank,
                    "unit": int(index),
                    "correlation": round(float(scan["correlations"][index]), 3),
                    "p_value": round(float(scan["p_values"][index]), 5),
                    "bonferroni_p": round(float(scan["bonferroni_adjusted"][index]), 5),
                    "fdr_p": round(float(scan["fdr_adjusted"][index]), 5),
                }
            )
        return rows

    def guardrail_rows(scan: dict[str, object], probe: dict[str, object]) -> list[dict[str, str]]:
        low, high = probe["test_accuracy_interval"]
        return [
            {
                "guardrail": "Multiple-comparison correction",
                "readout": f"raw {scan['raw_significant']}, FDR {scan['fdr_significant']}, Bonferroni {scan['bonferroni_significant']}",
                "why_it_matters": "If the raw hit count collapses after correction, the original story was probably driven by search over many units.",
            },
            {
                "guardrail": "Search-matched null model",
                "readout": f"best probe {pct(probe['best_accuracy'])} vs null 95th pct {pct(probe['null_95th_percentile'])}",
                "why_it_matters": "A picked winner only earns trust if it beats what the same search budget produces under shuffled labels.",
            },
            {
                "guardrail": "Uncertainty on the reported winner",
                "readout": f"95% bootstrap interval {pct(low)} to {pct(high)}",
                "why_it_matters": "A wide interval that still overlaps chance means the apparent explanation is too unstable to trust.",
            },
        ]

    def verdict_message(scan: dict[str, object], probe: dict[str, object]) -> tuple[str, str]:
        low, _ = probe["test_accuracy_interval"]
        if probe["best_accuracy"] <= probe["null_95th_percentile"]:
            return (
                "warn",
                f"""
**The picked winner still looks like a dead salmon.**

You can get a probe this good just by searching the same number of frozen random representations under shuffled labels.
The notebook is telling you that the *search procedure* is explaining the result better than the representation is.
""",
            )
        if low <= 0.5:
            return (
                "info",
                f"""
**The effect survives the null test, but the estimate is still fragile.**

Your best-picked probe clears the search-matched null, yet its uncertainty interval still overlaps chance.
This is the paper's broader point: even without an obvious false positive, explanations can stay high-variance and hard to trust.
""",
            )
        return (
            "success",
            f"""
**This regime is starting to look credible.**

The searched probe beats the search-matched null and its uncertainty interval clears chance.
That doesn't prove the explanation is *the* correct story, but it does mean the notebook's basic guardrails are finally aligned with a real planted signal.
""",
        )

    return (
        build_feature_figure,
        build_probe_figure,
        guardrail_rows,
        pct,
        scenario_name,
        top_feature_rows,
        verdict_message,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # The Dead Salmon Lab

        [*The Dead Salmons of AI Interpretability*](https://arxiv.org/abs/2512.18792) warns that interpretability can produce convincing-looking stories from random representations.

        This notebook turns that warning into a fast, CPU-only lab: instead of reproducing the paper's random BERT setup directly, it recreates the same failure mode with a frozen random MLP on synthetic data.

        The interactive question is simple: **when does a discovery stop being a dead salmon and start surviving the right guardrails?**

        You will watch three regimes:

        - **Null:** search produces impressive-looking hits even when everything is fake
        - **Weak signal:** there is some real structure, but the story is still fragile
        - **Real signal:** correction, search-matched nulls, and uncertainty begin to separate signal from search artifacts

        The custom extension is the **signal-strength slider**. Set it to `0.0` for a pure dead-salmon regime, then raise it to see when the exact same guardrails begin to earn belief.
        """
    )
    return


@app.cell
def _(mo):
    sample_count = mo.ui.slider(
        80,
        240,
        20,
        value=180,
        show_value=True,
        include_input=True,
        label="Sample count",
    )
    hidden_width = mo.ui.slider(
        32,
        192,
        16,
        value=96,
        show_value=True,
        include_input=True,
        label="Hidden units",
    )
    depth = mo.ui.slider(
        1,
        3,
        1,
        value=2,
        show_value=True,
        label="MLP depth",
    )
    probe_trials = mo.ui.slider(
        4,
        20,
        2,
        value=10,
        show_value=True,
        include_input=True,
        label="Probe search budget",
    )
    signal_strength = mo.ui.slider(
        0.0,
        2.0,
        0.2,
        value=0.0,
        show_value=True,
        include_input=True,
        label="Planted signal strength",
    )
    seed = mo.ui.number(
        0,
        9999,
        step=1,
        value=13,
        debounce=True,
        label="Random seed",
    )
    return depth, hidden_width, probe_trials, sample_count, seed, signal_strength


@app.cell(hide_code=True)
def _(depth, hidden_width, mo, probe_trials, sample_count, seed, signal_strength):
    controls = mo.vstack(
        [
            mo.md("## Controls"),
            mo.hstack([sample_count, hidden_width], widths="equal", gap=1.0),
            mo.hstack([depth, probe_trials], widths="equal", gap=1.0),
            mo.hstack([signal_strength, seed], widths="equal", gap=1.0),
        ],
        gap=0.9,
    )
    note = mo.callout(
        mo.md(
            """
            **How to use this lab**

            1. Start with `signal strength = 0.0` and notice how easy it is to find "interesting" units anyway.
            2. Increase the probe search budget to see selection effects inflate the best result.
            3. Raise the signal slider and watch correction, permutation tests, and uncertainty start telling a more useful story.
            """
        ),
        kind="info",
    )
    mo.hstack([controls, note], widths=[1.25, 0.95], align="start", gap=1.1)
    return


@app.cell
def _(
    build_judge_summary,
    depth,
    generate_dataset,
    hidden_width,
    probe_trials,
    random_mlp_features,
    run_probe_hunt,
    sample_count,
    seed,
    signal_strength,
    summarize_feature_scan,
):
    base_seed = int(seed.value or 0)
    dataset = generate_dataset(
        n_samples=int(sample_count.value),
        input_dim=12,
        signal_strength=float(signal_strength.value),
        seed=base_seed,
    )
    features = random_mlp_features(
        dataset["X"],
        hidden_width=int(hidden_width.value),
        depth=int(depth.value),
        seed=base_seed + 101,
    )
    scan = summarize_feature_scan(features, dataset["y"])
    probe = run_probe_hunt(
        dataset["X"],
        dataset["y"],
        hidden_width=int(hidden_width.value),
        depth=int(depth.value),
        probe_trials=int(probe_trials.value),
        permutation_trials=16,
        seed=base_seed + 1001,
    )
    judge_summary = build_judge_summary(scan, probe)
    return dataset, features, judge_summary, probe, scan


@app.cell(hide_code=True)
def _(judge_summary, mo, pct, probe, scenario_name, signal_strength):
    mo.md(
        f"""
        ## Scenario

        **{scenario_name(float(signal_strength.value))}**

        In this run, the best-picked probe reaches **{pct(probe["best_accuracy"])}** held-out accuracy.
        The current readout lands in the **{judge_summary["regime"].replace("_", " ")}** regime.
        The question is not "can I find something that looks compelling?" but "**does it still look compelling after I model the search process itself?**"
        """
    )
    return


@app.cell(hide_code=True)
def _(judge_summary, mo, pct, probe, probe_trials, scan):
    kind = {
        "null": "warn",
        "weak_signal": "info",
        "real_signal": "success",
    }[judge_summary["regime"]]
    mo.callout(
        mo.md(
            f"""
            **Judge summary: {judge_summary["headline"]}**

            {judge_summary["summary_line"]}

            Raw feature hits: **{judge_summary["raw_hits"]}**.
            FDR survivors: **{judge_summary["fdr_hits"]}**.
            The best-looking probe only matters if it beats the search-matched null.
            """
        ),
        kind=kind,
    )

    stats = mo.hstack(
        [
            mo.stat(
                value=judge_summary["regime"].replace("_", " "),
                label="measured regime",
                caption="based on guardrails, not the slider",
                bordered=True,
            ),
            mo.stat(
                value=f"{scan['raw_significant']} / {len(scan['p_values'])}",
                label="raw hits",
                caption="looks exciting before correction",
                bordered=True,
            ),
            mo.stat(
                value=str(scan["fdr_significant"]),
                label="FDR survivors",
                caption="what still survives adjustment",
                bordered=True,
            ),
            mo.stat(
                value=pct(probe["best_accuracy"]),
                label="best-picked probe",
                caption=f"after searching {int(probe_trials.value)} random probes",
                bordered=True,
            ),
            mo.stat(
                value=pct(probe["null_95th_percentile"]),
                label="null 95th percentile",
                caption="same search budget, shuffled labels",
                bordered=True,
            ),
            mo.stat(
                value=f"{probe['permutation_p_value']:.2f}",
                label="permutation p-value",
                caption="search-matched null test",
                bordered=True,
            ),
        ],
        wrap=True,
        gap=0.9,
    )
    stats
    return


@app.cell
def _(build_feature_figure, scan):
    feature_figure = build_feature_figure(scan)
    return (feature_figure,)


@app.cell(hide_code=True)
def _(feature_figure):
    feature_figure
    return


@app.cell(hide_code=True)
def _(mo, scan, top_feature_rows):
    table = mo.ui.table(
        top_feature_rows(scan),
        selection=None,
        pagination=False,
        page_size=8,
        show_download=False,
        label="Most suspicious hidden units",
    )
    mo.vstack(
        [
            mo.md(
                """
                ### What the feature scan is doing

                Each hidden unit gets a simple Pearson correlation test against the label.
                Under the null regime, *every single one of these units is spurious*.
                The table below shows how convincing some of them can still look before correction, which is exactly the dead-salmon trap.
                """
            ),
            table,
        ],
        gap=0.8,
    )
    return


@app.cell
def _(build_probe_figure, probe):
    probe_figure = build_probe_figure(probe)
    return (probe_figure,)


@app.cell(hide_code=True)
def _(probe_figure):
    probe_figure
    return


@app.cell(hide_code=True)
def _(guardrail_rows, mo, probe, scan, verdict_message):
    verdict_kind, verdict = verdict_message(scan, probe)
    guardrail_table = mo.ui.table(
        guardrail_rows(scan, probe),
        selection=None,
        pagination=False,
        show_download=False,
        label="Practical guardrails",
    )
    mo.vstack(
        [
            mo.md(
                """
                ## Guardrail readout

                The paper's practical claim is that interpretability findings should be treated like statistical estimates:
                define the estimand, compare it against meaningful alternatives, and quantify uncertainty before trusting the story.
                """
            ),
            mo.callout(mo.md(verdict), kind=verdict_kind),
            guardrail_table,
        ],
        gap=0.9,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Takeaway

        If a frozen random network can produce your interpretability story under the same search procedure, the story itself has not yet earned belief.

        A practical checklist for future interpretability work:

        1. **Model the search process**, not just the final chosen result.
        2. **Correct for multiple comparisons** when scanning many units, features, or subspaces.
        3. **Use search-matched nulls** such as permutations or randomized representations.
        4. **Report uncertainty**, not just a point estimate and a hand-picked visualization.
        5. **Check robustness across regimes** so the method can distinguish null structure from real signal.

        The broader lesson is not "never trust interpretability."
        It is: **trust the explanation only after it beats the same search process under the right null and uncertainty checks.**
        """
    )
    return


if __name__ == "__main__":
    app.run()
