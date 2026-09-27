"""Export evaluation figures from recorded aggregates, without retuning models."""

from pathlib import Path
from typing import Any


def calibration_figure(report: dict[str, Any], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = report["test"]["calibration_bins"]
    figure, (axis, counts) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    x = [item["mean_probability"] for item in bins]
    y = [item["observed_late_rate"] for item in bins]
    axis.plot([0, 1], [0, 1], "--", color="#7c8798", label="Perfect calibration")
    axis.plot(x, y, "o-", color="#1967b3", label="Released model, supplied test")
    axis.set(ylabel="Observed fraction with delay > 120 s", ylim=(-0.03, 1.03), xlim=(0, 1),
             title=f"Late-event probability calibration · Brier {report['test']['brier_score']:.3f}")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.2)
    counts.bar([(item["lower"] + item["upper"]) / 2 for item in bins], [item["rows"] for item in bins],
               width=0.085, color="#1967b3", alpha=0.7)
    counts.set(xlabel="Predicted probability", ylabel="Rows")
    figure.savefig(output, dpi=160)
    plt.close(figure)
