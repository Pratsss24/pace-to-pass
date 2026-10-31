"""Six figures, written straight to PNG so the README renders on GitHub."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

import passing                           # noqa: E402

INK = "#16181d"
MUTED = "#5b6472"
GRID = "#e3e6ea"
GOOD = "#17803d"
BAD = "#b4232c"
LINE = "#1f5fa8"
FILL = "#cfe0f2"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 12, "figure.dpi": 130, "savefig.bbox": "tight"})


def _frame(axis, title: str, subtitle: str = "") -> None:
    axis.set_title(title, loc="left", pad=22 if subtitle else 8,
                   fontweight="bold")
    if subtitle:
        axis.text(0, 1.02, subtitle, transform=axis.transAxes,
                  fontsize=9, color=MUTED, va="bottom")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def dose(path: Path, bands, fit, mean_gap, threshold) -> Path:
    """The raw bands, with the fitted curve laid over them."""
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    centres = [(band["low"] + band["high"]) / 2 for band in bands]
    centres[0] = bands[0]["high"] - 0.35
    centres[-1] = bands[-1]["low"] + 0.35
    shares = [100 * band["share"] for band in bands]
    errors = [100 * 1.96 * band["error"] for band in bands]

    grid = np.linspace(-1.6, 2.2, 240)
    curve = [100 * passing.episode_chance(fit, value, gap=mean_gap)
             for value in grid]
    axis.plot(grid, curve, color=LINE, linewidth=2.2,
              label="fitted", zorder=2)
    axis.errorbar(centres, shares, yerr=errors, fmt="o", color=INK,
                  markersize=6, capsize=4, linewidth=1.4, zorder=3,
                  label="measured, by band")

    axis.axhline(50, color=MUTED, linewidth=0.9, linestyle=":")
    axis.axvline(threshold, color=BAD, linewidth=1.4, linestyle="--")
    axis.annotate(f"{threshold:+.2f} s a lap\nfor an even chance",
                  xy=(threshold, 50), xytext=(threshold + 0.18, 22),
                  color=BAD, fontsize=9,
                  arrowprops=dict(arrowstyle="->", color=BAD, lw=1.1))
    axis.axvline(0, color=MUTED, linewidth=0.8)

    _frame(axis, "How much quicker you have to be",
           "chance an episode of following ends with the chasing car in "
           "front")
    axis.set_xlabel("free-air pace advantage of the chasing car, seconds "
                    "a lap")
    axis.set_ylabel("got past, %")
    axis.set_ylim(0, 100)
    axis.legend(frameon=False, loc="upper left", fontsize=9)
    out = path / "dose.png"
    figure.savefig(out)
    plt.close(figure)
    return out


def air(path: Path, air_table) -> Path:
    """Dirty air, split by whether the car behind was actually quicker."""
    figure, axis = plt.subplots(figsize=(7.0, 3.6))
    labels = ["held up by a car\nit was quicker than",
              "near a car it was\nnot quicker than", "clear air\n(the fit)"]
    values = [air_table["penalty when held up"],
              air_table["penalty when merely near"], 0.0]
    counts = [air_table["laps held up"], air_table["laps near but not quicker"],
              air_table["laps in clear air"]]
    colours = [BAD, MUTED, GOOD]
    bars = axis.bar(labels, values, color=colours, width=0.55)
    for bar, value, count in zip(bars, values, counts):
        axis.text(bar.get_x() + bar.get_width() / 2,
                  value + 0.012, f"{value:+.3f} s\n{count:,} laps",
                  ha="center", va="bottom", fontsize=9)
    axis.axhline(0, color=INK, linewidth=1.0)
    _frame(axis, "What a lap behind another car costs",
           "measured against the car's own free-air pace, not the field's")
    axis.set_ylabel("seconds a lap lost")
    axis.set_ylim(-0.04, max(values) * 1.45)
    out = path / "air.png"
    figure.savefig(out)
    plt.close(figure)
    return out


def guards(path: Path) -> Path:
    """What refusing to extrapolate did to the central estimate."""
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    labels = ["as first written",
              "implausible\nadvantages refused",
              "and no\nextrapolation"]
    values = [0.203, 0.958, 1.300]
    low = [0.024, 0.812, 1.003]
    high = [0.382, 1.104, 1.597]
    colours = [BAD, MUTED, GOOD]
    axis.errorbar(labels, values,
                  yerr=[np.array(values) - np.array(low),
                        np.array(high) - np.array(values)],
                  fmt="o", color=INK, markersize=0, capsize=5, linewidth=1.3,
                  zorder=2)
    axis.scatter(labels, values, color=colours, s=110, zorder=3)
    for index, value in enumerate(values):
        axis.annotate(f"{value:+.2f}", xy=(index, value),
                      xytext=(10, -4), textcoords="offset points",
                      fontsize=9, color=INK)
    _frame(axis, "A fifth of the rows were flattening the answer",
           "per-lap hazard slope, seconds a lap of advantage")
    axis.set_ylabel("log-odds per second a lap")
    axis.set_ylim(0, 1.8)
    axis.text(0.02, 0.92,
              "lines read outside the laps they were fitted on\n"
              "returned pace advantages of twenty-five seconds a lap",
              transform=axis.transAxes, fontsize=8.5, color=MUTED,
              va="top")
    out = path / "guards.png"
    figure.savefig(out)
    plt.close(figure)
    return out


def hazard_curve(path: Path, hazard) -> Path:
    """The per-lap chance, and how it falls away with time."""
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    steps = np.arange(1, 9)
    for value, colour, style in ((1.5, GOOD, "-"), (0.5, LINE, "-"),
                                 (0.0, MUTED, "--"), (-1.0, BAD, "-")):
        chances = []
        for step in steps:
            row = np.zeros(len(hazard.names))
            row[hazard.index("base")] = 1.0
            row[hazard.index("per second per lap of advantage")] = value
            row[hazard.index("per lap of trying")] = float(step)
            if "per second of opening gap" in hazard.names:
                row[hazard.index("per second of opening gap")] = 1.24
            import linear
            chances.append(100 * linear.probability(hazard, row))
        axis.plot(steps, chances, style, color=colour, linewidth=2.0,
                  marker="o", markersize=4,
                  label=f"{value:+.1f} s a lap quicker")
    _frame(axis, "Attempts do not accumulate",
           "chance of getting past on this lap, given this many laps spent "
           "behind")
    axis.set_xlabel("laps already spent directly behind")
    axis.set_ylabel("chance of passing on this lap, %")
    axis.legend(frameon=False, fontsize=9)
    out = path / "hazard.png"
    figure.savefig(out)
    plt.close(figure)
    return out


def circuits(path: Path, table, shuffle) -> Path:
    """Per-circuit thresholds, and the spread noise produces on its own."""
    named = [row for row in table if row["named"]
             and row["threshold"] is not None]
    figure, axis = plt.subplots(figsize=(7.6, 5.4))
    names = [row["circuit"] for row in named]
    values = [row["threshold"] for row in named]
    lows = [row["threshold"] - (row["low"] if row["low"] is not None
                                else row["threshold"]) for row in named]
    highs = [(row["high"] if row["high"] is not None else row["threshold"])
             - row["threshold"] for row in named]
    spots = np.arange(len(named))
    axis.errorbar(values, spots, xerr=[lows, highs], fmt="o", color=INK,
                  markersize=6, capsize=3, linewidth=1.2)
    axis.set_yticks(spots)
    axis.set_yticklabels(names)
    axis.invert_yaxis()
    axis.axvline(0, color=MUTED, linewidth=0.8)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.grid(axis="y", visible=False)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_title("A ranking that is not one", loc="left", pad=16,
                   fontweight="bold")
    axis.text(0, 1.02,
              f"threshold per circuit; shuffling the labels gives the same "
              f"spread (p {shuffle['p']:.2f})",
              transform=axis.transAxes, fontsize=9, color=MUTED,
              va="bottom")
    axis.set_xlabel("seconds a lap needed for an even chance of getting past")
    out = path / "circuits.png"
    figure.savefig(out)
    plt.close(figure)
    return out


def reliability(path: Path, trust, timing) -> Path:
    """Two checks side by side: the dose's noise, and the time a pass takes."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.8, 4.1))
    figure.subplots_adjust(wspace=0.28)

    parts = [trust["signal sd"] ** 2, trust["noise sd"] ** 2]
    total = sum(parts)
    left.barh([0], [100 * parts[0] / total], color=GOOD, height=0.5)
    left.barh([0], [100 * parts[1] / total], left=[100 * parts[0] / total],
              color=BAD, height=0.5)
    left.text(100 * parts[0] / total / 2, 0,
              f"signal\n{100 * parts[0] / total:.0f}%", ha="center",
              va="center", color="white", fontsize=10, fontweight="bold")
    left.text(100 * parts[0] / total + 100 * parts[1] / total / 2, 0,
              f"noise\n{100 * parts[1] / total:.0f}%", ha="center",
              va="center", color="white", fontsize=9.5)
    left.set_yticks([])
    left.set_xlim(0, 100)
    left.set_ylim(-1.1, 1.1)
    left.set_xlabel("share of the variance in the measured advantage")
    left.set_title("The dose is measured, not observed", loc="left",
                   pad=24, fontweight="bold")
    left.text(0, 1.03,
              f"odd clear laps against even ones, r = "
              f"{trust['correlation']:.2f}",
              transform=left.transAxes, fontsize=9, color=MUTED, va="bottom")
    for side in ("top", "right", "left"):
        left.spines[side].set_visible(False)

    bars = right.bar(["pure pace\nsays", "actually\ntakes"],
                     [timing["predicted"], timing["observed"]],
                     color=[MUTED, BAD], width=0.5)
    for bar, value in zip(bars, (timing["predicted"], timing["observed"])):
        right.text(bar.get_x() + bar.get_width() / 2, value + 0.05,
                   f"{value:.2f} laps", ha="center", va="bottom", fontsize=9)
    _frame(right, "A pass takes longer than pace says",
           f"median over {timing['n']} passes; {timing['ratio']:.2f} times "
           f"as long")
    right.set_ylabel("laps from first being within reach")
    right.set_ylim(0, timing["observed"] * 1.35)
    out = path / "checks.png"
    figure.savefig(out)
    plt.close(figure)
    return out


def draw(path: Path, **kwargs) -> dict[str, Path]:
    path.mkdir(parents=True, exist_ok=True)
    return {
        "dose": dose(path, kwargs["bands"], kwargs["fit"],
                     kwargs["mean_gap"], kwargs["threshold"]),
        "air": air(path, kwargs["air"]),
        "guards": guards(path),
        "hazard": hazard_curve(path, kwargs["hazard"]),
        "circuits": circuits(path, kwargs["table"], kwargs["shuffle"]),
        "checks": reliability(path, kwargs["trust"], kwargs["timing"]),
    }
