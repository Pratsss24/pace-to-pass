"""How quick each car actually is, measured where nothing is in its way.

The dose in this project is one car's pace advantage over another. It
cannot be read off the laps the two cars spend together, because those
laps are the outcome: a car stuck behind another is slowed by being stuck
behind it, so the pace difference measured in traffic is attenuated by
exactly the effect the project is trying to measure. Using it would be
circular, and the circularity runs in the direction that makes passing
look harder than it is.

So pace is measured apart and tested together. For each car-race, a line
is fitted through the laps it ran in clear air:

    seconds  =  base  +  wear * age  +  burn * lap

`age` is laps on the current set of tyres and `lap` carries fuel burn and
track evolution, the same two-term shape the earlier projects used. The
fit is over that car's own laps only, so a car's base pace absorbs its
machinery, its driver and the day, and nothing has to be assumed about
how those compare across teams.

The **advantage** of one car over another on a given lap is then the
difference of the two fitted lines, each evaluated at that car's own tyre
age on that lap. Fuel and tyres are therefore held level by construction
rather than by a covariate, which matters here: a car that has just
stopped is quicker than the car ahead for reasons that have nothing to do
with whether it can pass.

**The dirty-air penalty** falls out of the same fit for free. The
difference between what a car actually ran while following and what its
own free-air line says it should have run is the cost of being there, in
seconds per lap, measured against the car's own pace rather than against
the field's.

A car-race with too few clear laps to support a three-parameter fit is
refused rather than extrapolated. Cars that spend a whole race in traffic
are exactly the ones whose free-air pace is least knowable, and they drop
out of the sample for a reason that is worth stating out loud.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

import stints
from data import Lap, Result, Stop

MIN_FREE_LAPS = 8     # clear laps needed before a car-race's line is trusted
MAX_ENTANGLED = 0.95  # correlation between age and lap the fit will accept
MAX_ADVANTAGE = 3.0   # seconds a lap: beyond this is a broken fit, not pace
MARGIN = 0            # laps of extrapolation allowed beyond the fitted range
COLUMNS = 3           # base, wear, burn

Key = tuple[int, int, str]


@dataclass(frozen=True)
class Line:
    """One car-race's pace, and the range of laps it was learnt from.

    The range is carried because a line fitted to laps 20 to 40 of a race
    says nothing trustworthy about lap 3. Inside a single stint tyre age
    and race lap move together, so the split between the two coefficients
    is partly arbitrary even when the fitted values are good, and reading
    the line outside its range turns that arbitrariness into a number.
    Doing it anyway produced pace advantages of twenty-five seconds a lap
    in this season, which is not a pace advantage but a broken fit.
    """

    beta: np.ndarray
    lap_lo: int
    lap_hi: int
    age_lo: int
    age_hi: int
    n: int

    def covers(self, lap: int, age: int, margin: int = MARGIN) -> bool:
        return (self.lap_lo - margin <= lap <= self.lap_hi + margin
                and self.age_lo - margin <= age <= self.age_hi + margin)

    def at(self, lap: int, age: int) -> float:
        return float(self.beta[0] + self.beta[1] * age + self.beta[2] * lap)


def ages(stops: list[Stop], laps: list[Lap]) -> dict[Key, dict[int, int]]:
    """Tyre age for every lap of every car-race."""
    table = stints.stop_laps(stops)
    out: dict[Key, dict[int, int]] = defaultdict(dict)
    for lap in laps:
        key = (lap.season, lap.round, lap.driver)
        _, age = stints.assign_stint(lap.lap, table.get(key, []))
        out[key][lap.lap] = age
    return dict(out)


def free_air(clean: list[stints.StintLap], traffic: dict[Key, set[int]],
             *, minimum: int = MIN_FREE_LAPS,
             entangled: float = MAX_ENTANGLED
             ) -> tuple[dict[Key, np.ndarray], dict[str, int]]:
    """Fit each car-race's pace on the laps it ran in clear air."""
    rows: dict[Key, list[stints.StintLap]] = defaultdict(list)
    held = 0
    for row in clean:
        if row.lap in traffic.get(row.key, ()):
            held += 1
            continue
        rows[row.key].append(row)

    census = {"clean laps": len(clean), "held out: in traffic": held,
              "refused: too few clear laps": 0,
              "refused: age and lap not separable": 0,
              "fitted car-races": 0}
    lines: dict[Key, np.ndarray] = {}
    for key, car in rows.items():
        if len(car) < minimum:
            census["refused: too few clear laps"] += 1
            continue
        design = np.column_stack([
            np.ones(len(car)),
            np.array([row.age for row in car], dtype=float),
            np.array([row.lap for row in car], dtype=float)])
        # Inside a single stint, tyre age and race lap advance together, so
        # the two coefficients are not separately identified and the split
        # between them is arbitrary. Fitted values stay fine across the laps
        # actually run, but the line is then being read at tyre ages this
        # car never had, and an arbitrary split extrapolates to nonsense.
        # One car-race in this season reached a predicted spread of twenty
        # seconds across the field before this guard was added.
        # This check comes before the rank check on purpose: a car-race
        # that never stopped has age equal to lap exactly, so the design
        # is rank deficient *because* the two are the same column, and
        # reporting that as a shortage of laps would send anyone reading
        # the census looking for the wrong problem.
        if abs(np.corrcoef(design[:, 1], design[:, 2])[0, 1]) > entangled:
            census["refused: age and lap not separable"] += 1
            continue
        if np.linalg.matrix_rank(design) < COLUMNS:
            census["refused: too few clear laps"] += 1
            continue
        target = np.array([row.seconds for row in car])
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        lines[key] = Line(beta=beta,
                          lap_lo=min(row.lap for row in car),
                          lap_hi=max(row.lap for row in car),
                          age_lo=min(row.age for row in car),
                          age_hi=max(row.age for row in car),
                          n=len(car))
        census["fitted car-races"] += 1
    return lines, census


def predict(lines: dict[Key, Line], key: Key, lap: int, age: int, *,
            margin: int = MARGIN) -> float:
    """What this car would have run on this lap with nothing in the way.

    NaN outside the range of laps and tyre ages the line was fitted on.
    Refusing to extrapolate costs about a fifth of the episodes and is
    the single change that most improves this project's central estimate.
    """
    line = lines.get(key)
    if line is None or not line.covers(lap, age, margin):
        return float("nan")
    return line.at(lap, age)


def advantage(lines: dict[Key, Line], age_table: dict[Key, dict],
              ahead: Key, behind: Key, lap: int, *,
              cap: float = MAX_ADVANTAGE, margin: int = MARGIN) -> float:
    """Seconds per lap the chasing car is quicker, in clear air.

    Positive means the car behind is the quicker of the two. Each car is
    evaluated at its own tyre age on this lap, so the number is a pace
    difference and not a tyre-state difference.
    """
    age_ahead = age_table.get(ahead, {}).get(lap)
    age_behind = age_table.get(behind, {}).get(lap)
    if age_ahead is None or age_behind is None:
        return float("nan")
    value = (predict(lines, ahead, lap, age_ahead, margin=margin)
             - predict(lines, behind, lap, age_behind, margin=margin))
    # Two adjacent cars in a Formula 1 race are not three seconds a lap
    # apart in clear air. A number past this is a fit that has failed,
    # and in a logit it is a high-leverage point that flattens the very
    # slope being estimated - here by a factor of five.
    if not np.isfinite(value) or abs(value) > cap:
        return float("nan")
    return value


def held_up_laps(traffic: dict[Key, dict[int, Key]],
                 lines: dict[Key, Line], age_table: dict[Key, dict],
                 *, threshold: float = 0.0) -> dict[Key, set[int]]:
    """Laps where a car sat behind somebody it was quicker than.

    These are the laps on which the deficit has to be the car in front.
    """
    out: dict[Key, set[int]] = defaultdict(set)
    for behind, table in traffic.items():
        for lap, ahead in table.items():
            edge = advantage(lines, age_table, ahead, behind, lap)
            if np.isfinite(edge) and edge > threshold:
                out[behind].add(lap)
    return dict(out)


def dirty_air(clean: list[stints.StintLap], lines: dict[Key, Line],
              traffic: dict[Key, set[int]], held_up: dict[Key, set[int]]
              ) -> dict:
    """What a lap spent behind another car costs, against the car's own line.

    Measured only on laps that were held out of the fit, so the line it is
    compared against never saw them.

    The headline is the *held up* figure: laps where the car behind was
    the quicker of the two in clear air. Those are the laps where the
    deficit has to be the car in front, because the car behind had the
    pace to go faster and did not. Pooling every lap within two seconds
    dilutes that badly - most such laps are cars running at a pace they
    were going to run anyway, sitting behind someone of similar speed.

    Residuals are summarised by their median. The fit's own residuals sum
    to zero by construction but lap times are right-skewed, so the median
    residual in clear air sits a little below zero; the comparison is
    between the two medians, not between a median and zero.
    """
    stuck: list[float] = []
    nearby: list[float] = []
    clear: list[float] = []
    for row in clean:
        expected = predict(lines, row.key, row.lap, row.age)
        if not np.isfinite(expected):
            continue
        excess = row.seconds - expected
        if row.lap in held_up.get(row.key, ()):
            stuck.append(excess)
        elif row.lap in traffic.get(row.key, ()):
            nearby.append(excess)
        else:
            clear.append(excess)

    def middle(values: list[float]) -> float:
        return float(np.median(values)) if values else float("nan")

    base = middle(clear)
    return {"laps held up": len(stuck),
            "laps near but not quicker": len(nearby),
            "laps in clear air": len(clear),
            "clear air residual": base,
            "penalty when held up": middle(stuck) - base,
            "penalty when merely near": middle(nearby) - base}


def spread(lines: dict[Key, Line]) -> dict:
    """A sanity check on the fitted lines: the field's own pace spread.

    Each car is read at the middle of its own fitted range, where no
    extrapolation is involved, and compared with the rest of its field at
    a common tyre age. A Formula 1 grid covers two to three seconds a lap.
    If these lines were mostly noise, this would be far wider, and before
    the range and plausibility guards went in it was.
    """
    races: dict[tuple[int, int], list[float]] = defaultdict(list)
    for key, line in lines.items():
        lap = (line.lap_lo + line.lap_hi) // 2
        age = (line.age_lo + line.age_hi) // 2
        value = line.at(lap, age)
        if np.isfinite(value):
            races[key[:2]].append(value)
    widths = [max(values) - min(values)
              for values in races.values() if len(values) > 4]
    return {"races": len(widths),
            "median spread": float(np.median(widths)) if widths else
            float("nan"),
            "widest": float(max(widths)) if widths else float("nan")}


def reliability(clean: list[stints.StintLap], traffic: dict,
                age_table: dict[Key, dict], episodes) -> dict:
    """How much of the measured pace advantage is signal, by split halves.

    The dose is measured, not observed, and a noisy dose flattens the
    slope fitted to it - classical attenuation. Fitting each car's line
    twice, once on its odd clear laps and once on its even ones, gives
    two independent measurements of the same quantity. The spread between
    them is twice the noise variance, which is enough to say how much of
    the variation in the dose is real and roughly how much the slope is
    being pulled toward zero.
    """
    odd = [row for row in clean if row.lap % 2]
    even = [row for row in clean if not row.lap % 2]
    first, _ = free_air(odd, traffic, minimum=MIN_FREE_LAPS // 2)
    second, _ = free_air(even, traffic, minimum=MIN_FREE_LAPS // 2)

    pairs = []
    for episode in episodes:
        ahead = (episode.season, episode.round, episode.ahead)
        behind = (episode.season, episode.round, episode.behind)
        one = advantage(first, age_table, ahead, behind, episode.start)
        two = advantage(second, age_table, ahead, behind, episode.start)
        if np.isfinite(one) and np.isfinite(two):
            pairs.append((one, two))
    if len(pairs) < 30:
        return {"n": len(pairs)}

    one = np.array([row[0] for row in pairs])
    two = np.array([row[1] for row in pairs])
    noise = float(np.var(one - two, ddof=1) / 2)
    measured = float(np.var(np.concatenate([one, two]), ddof=1))
    signal = max(measured - noise, 0.0)
    return {"n": len(pairs),
            "correlation": float(np.corrcoef(one, two)[0, 1]),
            "noise sd": float(np.sqrt(noise)),
            "signal sd": float(np.sqrt(signal)),
            "reliability": signal / measured if measured else float("nan")}
