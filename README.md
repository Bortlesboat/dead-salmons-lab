# The Dead Salmons Lab

An interactive marimo notebook based on [*The Dead Salmons of AI Interpretability*](https://arxiv.org/abs/2512.18792).

This notebook turns the paper's core warning into a fast, CPU-friendly demo: random representations can still produce convincing-looking interpretability stories if you search hard enough. Instead of reproducing the paper's random-BERT setup directly, it recreates the same failure mode with a frozen random MLP on synthetic data.

The custom extension is the **signal-strength slider**. Start in a pure null regime, then raise real signal and watch when multiple-testing correction, search-matched nulls, and uncertainty finally begin to separate genuine structure from dead-salmon artifacts.

## Files

- `the_dead_salmons_lab.py`: the public competition notebook
- `dead_salmons_lab.py`: the tested simulation core used by the local workspace version

## Live Notebook

- `molab` link: https://molab.marimo.io/github/Bortlesboat/dead-salmons-lab/blob/main/the_dead_salmons_lab.py

## Submission Blurb

> This notebook turns *The Dead Salmons of AI Interpretability* into a fast interactive lab. It recreates the paper's false-discovery failure mode with a frozen random MLP on synthetic data, then adds a planted-signal slider to show when multiple-testing correction, search-matched nulls, and uncertainty estimation finally begin to separate real signal from dead-salmon artifacts.
