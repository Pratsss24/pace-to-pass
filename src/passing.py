"""The chance of getting past, per lap of trying.

Every lap a car spends directly behind another is a trial that either ends
in a pass or does not. That is a discrete-time hazard, and it is the right
shape for this question for two reasons.

The first is censoring. Plenty of episodes end because the race ended, or
because somebody pitted, and the car might well have got past on the next
lap. Scoring those as failures would be wrong, and dropping them would
throw away the laps they did contribute. A hazard model keeps the laps and
stays silent about the lap that never happened.

The second is that episode length is an outcome, not a covariate. A pass
on the second lap of following makes the episode two laps long; putting
"laps spent following" into an episode-level model as an explanatory
variable regresses the outcome on itself. In the hazard it is time, which
is what it actually is.

    h(t)  =  logistic( base + edge * advantage + wear * t + circuit )

`advantage` is the chasing car's free-air pace advantage in seconds per
lap, measured by `pace` on laps the two cars did not spend together. The
circuit term is an intercept shift: how much harder this track is than
the average, holding the pace difference level. That shift, converted
back into seconds per lap, is the number this project is for.

**Reading the coefficients.** They are log-odds and they do not mean much
directly. `chance` turns them into the probability of getting past within
a window of laps, and `threshold` inverts that: the pace advantage a car
needs for an even chance of clearing the car in front within the window.
Seconds per lap is a unit anyone can argue with, which is the point.

**Clustering.** The laps of one episode are one observation seen several
times, so errors are clustered on the episode.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

import linear
import pace
from follow import Episode

WINDOW = 5            # laps: the span a threshold is quoted over
MIN_EPISODES = 25     # episodes a circuit needs before it gets its own term
MIN_PASSES = 3        # and passes, or its intercept is not identified
SEARCH = (-6.0, 6.0)  # seconds per lap: the range a threshold is sought in
BISECTIONS = 60


def trials(episodes: list[Episode], lines: dict, age_table: dict) -> dict:
    """One row per lap of following, with the lap's outcome and dose.

    The advantage is evaluated at the episode's first lap and held fixed
    across it. It is a property of the two cars, not of the lap, and
    letting it move with tyre age inside the episode would smuggle the
    tyre state back into a quantity built to be free of it.
    """
    rows: list[tuple] = []
    kept: list[Episode] = []
    dropped = 0
    for episode in episodes:
        ahead = (episode.season, episode.round, episode.ahead)
        behind = (episode.season, episode.round, episode.behind)
        edge = pace.advantage(lines, age_table, ahead, behind, episode.start)
        if not np.isfinite(edge):
            dropped += 1
            continue
        kept.append(episode)
        for step in range(1, episode.laps + 1):
            last = step == episode.laps
            rows.append((1.0 if (last and episode.passed) else 0.0,
                         edge, float(step), episode.gap_start,
                         episode.circuit, episode.name))

    if not rows:
        raise ValueError("no episode had a measurable pace advantage")
    outcome = np.array([row[0] for row in rows])
    labels = [row[5] for row in rows]
    index = {name: number for number, name in enumerate(sorted(set(labels)))}
    return {"passed": outcome,
            "advantage": np.array([row[1] for row in rows]),
            "lap": np.array([row[2] for row in rows]),
            "gap": np.array([row[3] for row in rows]),
            "circuit": np.array([row[4] for row in rows]),
            "group": np.array([index[name] for name in labels]),
            "episodes": kept,
            "dropped: no free-air pace for one of the cars": dropped}


def circuit_terms(rows: dict, *, minimum: int = MIN_EPISODES,
                  passes: int = MIN_PASSES) -> tuple[str, ...]:
    """Circuits with enough episodes, and enough passes, to carry a term.

    A circuit where nobody got past has no finite intercept of its own -
    the fit would separate - so it stays in the reference group and is
    reported as a circuit the season cannot resolve rather than as a
    circuit where passing is infinitely hard.
    """
    counted: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for circuit, outcome in zip(rows["circuit"], rows["passed"]):
        counted[circuit][0] += 1
        counted[circuit][1] += int(outcome)
    return tuple(sorted(
        name for name, (episodes, scored) in counted.items()
        if episodes >= minimum and scored >= passes))


def design(arrays: dict, circuits: tuple[str, ...], *, gap: bool = True
           ) -> tuple[np.ndarray, tuple[str, ...]]:
    """The hazard design: pace, time, optionally the opening gap, circuits."""
    n = arrays["passed"].size
    columns = [np.ones(n), arrays["advantage"], arrays["lap"]]
    names = ["base", "per second per lap of advantage", "per lap of trying"]
    if gap:
        columns.append(arrays["gap"])
        names.append("per second of opening gap")
    for circuit in circuits:
        columns.append((arrays["circuit"] == circuit).astype(float))
        names.append(f"circuit: {circuit}")
    return np.column_stack(columns), tuple(names)


def hazard(arrays: dict, circuits: tuple[str, ...] = (), *,
           gap: bool = True) -> linear.Fit:
    """Fit the per-lap chance of getting past."""
    matrix, names = design(arrays, circuits, gap=gap)
    return linear.logit(arrays["passed"], matrix, names, arrays["group"],
                        label="chance of passing on a given lap")


def chance(fit: linear.Fit, advantage: float, *, circuit: str = "",
           window: int = WINDOW, gap: float = 1.0) -> float:
    """The chance of getting past within `window` laps of following.

    One minus the chance of failing on every lap of the window. The
    per-lap hazard rises or falls with `t`, so the laps are not
    exchangeable and the product is taken term by term.
    """
    survive = 1.0
    for step in range(1, window + 1):
        row = np.zeros(len(fit.names))
        row[fit.index("base")] = 1.0
        row[fit.index("per second per lap of advantage")] = advantage
        row[fit.index("per lap of trying")] = float(step)
        if "per second of opening gap" in fit.names:
            row[fit.index("per second of opening gap")] = gap
        name = f"circuit: {circuit}"
        if name in fit.names:
            row[fit.index(name)] = 1.0
        survive *= 1.0 - linear.probability(fit, row)
    return 1.0 - survive


def threshold(fit: linear.Fit, *, circuit: str = "", window: int = WINDOW,
              target: float = 0.5, gap: float = 1.0) -> float | None:
    """The pace advantage giving an even chance of getting past.

    Bisection, which is safe here because the chance is monotone in the
    advantage: the coefficient is positive, every lap's hazard rises with
    it, and a product of rising terms rises. Returns None when the whole
    search range sits on one side of the target, which is the honest
    answer for a circuit where even a large advantage is not enough.
    """
    low, high = SEARCH
    if chance(fit, low, circuit=circuit, window=window, gap=gap) > target:
        return None
    if chance(fit, high, circuit=circuit, window=window, gap=gap) < target:
        return None
    for _ in range(BISECTIONS):
        middle = (low + high) / 2
        if chance(fit, middle, circuit=circuit, window=window,
                  gap=gap) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def episodes_of(arrays: dict) -> dict:
    """Collapse the lap rows back to one row per episode.

    The hazard needs the laps; the question "how quick do you have to be
    to get past this car" is about the episode. Both are fitted, and they
    have to agree, because they are two summaries of the same rows.
    """
    seen: dict = {}
    for edge, gap, circuit, group in zip(
            arrays["advantage"], arrays["gap"], arrays["circuit"],
            arrays["group"]):
        seen.setdefault(group, {"advantage": edge, "gap": gap,
                                "circuit": circuit, "laps": 0, "passed": 0})
    for group, step, outcome in zip(arrays["group"], arrays["lap"],
                                    arrays["passed"]):
        seen[group]["laps"] = max(seen[group]["laps"], int(step))
        seen[group]["passed"] = max(seen[group]["passed"], int(outcome))

    order = sorted(seen)
    return {"passed": np.array([float(seen[g]["passed"]) for g in order]),
            "advantage": np.array([seen[g]["advantage"] for g in order]),
            "gap": np.array([seen[g]["gap"] for g in order]),
            "laps": np.array([float(seen[g]["laps"]) for g in order]),
            "circuit": np.array([seen[g]["circuit"] for g in order]),
            "group": np.array(order)}


def episode_design(rows: dict, circuits: tuple[str, ...]
                   ) -> tuple[np.ndarray, tuple[str, ...]]:
    """Pace advantage and opening gap, plus a term per named circuit.

    Deliberately no term for how long the episode lasted. That is an
    outcome - a pass on the second lap makes the episode two laps long -
    and putting it on the right-hand side would be regressing the outcome
    on itself.
    """
    n = rows["passed"].size
    columns = [np.ones(n), rows["advantage"], rows["gap"]]
    names = ["base", "per second per lap of advantage",
             "per second of opening gap"]
    for circuit in circuits:
        columns.append((rows["circuit"] == circuit).astype(float))
        names.append(f"circuit: {circuit}")
    return np.column_stack(columns), tuple(names)


def episode_fit(rows: dict, circuits: tuple[str, ...] = ()) -> linear.Fit:
    """Whether an episode ends in a pass, against the pace difference."""
    matrix, names = episode_design(rows, circuits)
    return linear.logit(rows["passed"], matrix, names, rows["group"],
                        label="whether this episode ends in a pass")


def episode_chance(fit: linear.Fit, advantage: float, *, circuit: str = "",
                   gap: float = 1.0) -> float:
    """The chance this episode ends with the chasing car in front."""
    row = np.zeros(len(fit.names))
    row[fit.index("base")] = 1.0
    row[fit.index("per second per lap of advantage")] = advantage
    row[fit.index("per second of opening gap")] = gap
    name = f"circuit: {circuit}"
    if name in fit.names:
        row[fit.index(name)] = 1.0
    return linear.probability(fit, row)


def episode_threshold(fit: linear.Fit, *, circuit: str = "",
                      target: float = 0.5, gap: float = 1.0) -> float | None:
    """The pace advantage at which a pass becomes more likely than not.

    This is the project's headline number, and it is a plain inversion of
    the fitted logit rather than a projection onto a window of laps the
    episode may never reach.
    """
    low, high = SEARCH
    if episode_chance(fit, low, circuit=circuit, gap=gap) > target:
        return None
    if episode_chance(fit, high, circuit=circuit, gap=gap) < target:
        return None
    for _ in range(BISECTIONS):
        middle = (low + high) / 2
        if episode_chance(fit, middle, circuit=circuit, gap=gap) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def by_circuit(fit: linear.Fit, circuits: tuple[str, ...], rows: dict, *,
               gap: float = 1.0) -> list[dict]:
    """Each circuit's threshold, hardest first, with its own interval.

    The interval comes from the circuit term's own standard error, moved
    onto the seconds scale by re-inverting at each end. A circuit term is
    an intercept shift in log-odds, and a shift of a fixed size is worth
    more seconds where the pace slope is shallow, so the interval is not
    symmetric and is not treated as if it were.
    """
    out = []
    for circuit in ("",) + tuple(circuits):
        name = f"circuit: {circuit}"
        here = rows["circuit"] == circuit if circuit else np.array(
            [one not in circuits for one in rows["circuit"]])
        value = episode_threshold(fit, circuit=circuit, gap=gap)
        low = high = None
        if circuit and name in fit.names:
            shift = fit[name]
            spread = fit.error(name)
            moved = []
            for end in (shift - 1.96 * spread, shift + 1.96 * spread):
                copy = linear.Fit(
                    names=fit.names,
                    beta=np.where(np.arange(len(fit.names))
                                  == fit.index(name), end, fit.beta),
                    covariance=fit.covariance, n=fit.n, groups=fit.groups,
                    kind=fit.kind)
                moved.append(episode_threshold(copy, circuit=circuit,
                                               gap=gap))
            low, high = min(moved, key=lambda v: (v is None, v)),                 max(moved, key=lambda v: (v is None, v or 0.0))
        out.append({"circuit": circuit or "the rest, pooled",
                    "threshold": value, "low": low, "high": high,
                    "shift": fit[name] if name in fit.names else 0.0,
                    "episodes": int(here.sum()),
                    "passes": int(rows["passed"][here].sum()),
                    "named": bool(circuit)})
    out.sort(key=lambda row: (row["threshold"] is None,
                              row["threshold"] or 0.0), reverse=True)
    return out


def dose_table(arrays: dict, edges=(-9.0, -1.0, -0.5, 0.0, 0.5, 1.0, 9.0)
               ) -> list[dict]:
    """The raw episode-level pass rate by band of pace advantage.

    No model in it. If the fitted hazard says something the bands do not,
    the model is wrong and not the other way round.
    """
    seen: dict = {}
    for edge, group, circuit in zip(arrays["advantage"], arrays["group"],
                                    arrays["circuit"]):
        seen.setdefault(group, [edge, 0])
    for group, outcome in zip(arrays["group"], arrays["passed"]):
        seen[group][1] = max(seen[group][1], int(outcome))

    values = np.array([row[0] for row in seen.values()])
    scored = np.array([row[1] for row in seen.values()], dtype=float)
    rows = []
    for low, high in zip(edges, edges[1:]):
        band = (values >= low) & (values < high)
        if not band.any():
            continue
        share = float(scored[band].mean())
        n = int(band.sum())
        rows.append({"low": low, "high": high, "n": n, "share": share,
                     "error": float(np.sqrt(share * (1 - share) / n))})
    return rows


def laps_to_pass(arrays: dict, *, minimum: float = 0.2) -> dict:
    """How long a pass takes, against how long pure pace says it should.

    A car `g` seconds behind and `a` seconds a lap quicker closes the gap
    in `g / a` laps and is ahead on the next one. That is the whole of the
    pure-pace prediction, and it ignores the car in front entirely. The
    ratio of what happened to what pace predicts is the resistance, in a
    unit with no model in it at all.

    Only episodes with a real advantage are counted: the prediction
    divides by it, and for a car that is barely quicker it is enormous
    and meaningless.
    """
    seen: dict = {}
    for edge, gap, group in zip(arrays["advantage"], arrays["gap"],
                                arrays["group"]):
        seen.setdefault(group, [edge, gap, 0, 0])
    for group, step, outcome in zip(arrays["group"], arrays["lap"],
                                    arrays["passed"]):
        seen[group][2] = max(seen[group][2], int(step))
        seen[group][3] = max(seen[group][3], int(outcome))

    predicted, observed = [], []
    never = 0
    for edge, gap, length, scored in seen.values():
        if edge < minimum:
            continue
        expected = gap / edge
        if scored:
            predicted.append(expected)
            observed.append(float(length))
        elif length > expected:
            never += 1
    if not predicted:
        return {"n": 0}
    predicted = np.array(predicted)
    observed = np.array(observed)
    return {"n": int(predicted.size),
            "predicted": float(np.median(predicted)),
            "observed": float(np.median(observed)),
            "ratio": float(np.median(observed / predicted)),
            "still behind past the predicted lap": never}


def circuit_spread(fit: linear.Fit, circuits: tuple[str, ...], rows: dict,
                   *, gap: float = 1.0) -> float:
    """The range of fitted thresholds across the named circuits."""
    values = [episode_threshold(fit, circuit=one, gap=gap)
              for one in circuits]
    values = [one for one in values if one is not None]
    return max(values) - min(values) if len(values) > 1 else float("nan")


def permute_circuits(rows: dict, circuits: tuple[str, ...], *,
                     draws: int = 400, gap: float = 1.0,
                     seed: int = 0) -> dict:
    """How wide a spread of circuit thresholds pure noise would produce.

    The fitted thresholds range over most of a second a lap, which looks
    like a lot until you ask what the same procedure returns when the
    circuit labels are meaningless. Shuffling the labels across episodes
    destroys any real circuit effect and keeps everything else - the
    sample size, the imbalance between circuits, the pace distribution -
    exactly as it is. The share of shuffles that beat the real spread is
    a p-value that assumes almost nothing.
    """
    observed = circuit_spread(episode_fit(rows, circuits), circuits, rows,
                              gap=gap)
    rng = np.random.default_rng(seed)
    shuffled = rows | {}
    beaten = 0
    kept: list[float] = []
    for _ in range(draws):
        shuffled["circuit"] = rng.permutation(rows["circuit"])
        try:
            fake = episode_fit(shuffled, circuits)
            width = circuit_spread(fake, circuits, shuffled, gap=gap)
        except ValueError:
            continue
        if not np.isfinite(width):
            continue
        kept.append(width)
        if width >= observed:
            beaten += 1
    return {"observed": observed, "draws": len(kept),
            "median under shuffling": float(np.median(kept)) if kept
            else float("nan"),
            "p": (beaten + 1) / (len(kept) + 1) if kept else float("nan")}
