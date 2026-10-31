"""Measure how much quicker a car has to be to get past the one in front.

Run with no arguments; the season is cached, so a second run costs nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import charts                                              # noqa: E402
import data                                                # noqa: E402
import follow                                              # noqa: E402
import gaps                                                # noqa: E402
import linear                                              # noqa: E402
import pace                                                # noqa: E402
import passing                                             # noqa: E402
import report                                              # noqa: E402
import stints                                              # noqa: E402

SEASONS = (2023,)
OUT = ROOT / "out"
DRAWS = 400

_log: list[str] = []


def say(line: str = "") -> None:
    print(line)
    _log.append(line)


def main() -> None:
    laps, stops, results = data.load(SEASONS)
    times = gaps.elapsed(laps)

    # ---- the instrument, checked against the feed's own positions -------
    agreement = follow.order_agreement(laps, times)
    say(f"{len(laps):,} lap times, {len(stops):,} pit stops")
    say()
    say("race order, derived by summing lap times, against the position "
        "the feed reports")
    say(f"  agree {agreement['agree']:,}   disagree "
        f"{agreement['disagree']:,}   {agreement['share']:.4%}")
    say("  so the gap between any two cars is recoverable exactly from "
        "lap times, and nothing downstream needs a positional feed")

    # ---- who was following whom ------------------------------------------
    episodes, census = follow.build(laps, stops, results)
    say()
    say(f"following episodes within {follow.REACH:.0f} s, directly behind")
    for key, value in sorted(census.items()):
        say(f"  {key:46s} {value:6d}")
    say(f"  {len(episodes)} episodes, "
        f"{sum(e.passed for e in episodes)} ending in a pass")

    # ---- how quick each car was, measured where nothing was in the way --
    traffic = follow.in_traffic(laps, times)
    clean, _ = stints.build(laps, stops, results)
    lines, pace_census = pace.free_air(clean, traffic)
    age_table = pace.ages(stops, laps)
    say()
    say("free-air pace, fitted per car-race on laps run in clear air")
    for key, value in sorted(pace_census.items()):
        say(f"  {key:46s} {value:6d}")
    field = pace.spread(lines)
    say(f"  the fitted field spans {field['median spread']:.2f} s a lap at "
        f"the median race, which is what a Formula 1 grid covers")

    held = pace.held_up_laps(traffic, lines, age_table)
    air = pace.dirty_air(clean, lines, traffic, held)
    say()
    say("what a lap spent behind another car costs, against the car's own "
        "free-air line")
    say(f"  held up by a car it was quicker than   "
        f"{air['penalty when held up']:+.3f} s  "
        f"({air['laps held up']:,} laps)")
    say(f"  near a car it was not quicker than     "
        f"{air['penalty when merely near']:+.3f} s  "
        f"({air['laps near but not quicker']:,} laps)")
    say("  the second number is the control: being close to a car costs "
        "nothing in itself. The cost appears only when the car behind had "
        "the pace to go faster and could not")

    # ---- the dose, and what it buys --------------------------------------
    arrays = passing.trials(episodes, lines, age_table)
    rows = passing.episodes_of(arrays)
    mean_gap = float(rows["gap"].mean())
    say()
    say(f"{rows['passed'].size} episodes carry a measurable pace "
        f"advantage; {arrays['dropped: no free-air pace for one of the cars']}"
        f" do not and are dropped")
    say("  a car-race's line is never read outside the laps and tyre ages "
        "it was fitted on, which is what most of that costs")

    say()
    say("how often an episode ends with the chasing car in front, by band "
        "of free-air pace advantage - no model in this table")
    say(f"  {'advantage':>18s} {'episodes':>9s} {'got past':>14s}")
    bands = passing.dose_table(arrays)
    for band in bands:
        label = f"{band['low']:+.1f} to {band['high']:+.1f}"
        say(f"  {label:>18s} {band['n']:9d} "
            f"{band['share']:12.1%} +/- {1.96 * band['error']:.1%}")

    fit = passing.episode_fit(rows)
    say()
    say(f"the same thing fitted (n={fit.n}, {fit.groups} clusters, "
        f"McFadden {fit.fit_statistic:.3f})")
    for name in fit.names:
        low, high = fit.interval(name)
        say(f"  {name:34s} {fit[name]:+8.4f} ({low:+.4f} to {high:+.4f})  "
            f"p {fit.p_value(name):.4f}")
    say(f"  the intercept is {fit['base']:+.3f}: two cars of identical pace "
        f"nose to tail are a coin toss, which is what it should be and was "
        f"not imposed")

    threshold = passing.episode_threshold(fit, gap=mean_gap)
    say()
    say(f"so, at the mean opening gap of {mean_gap:.2f} s, a car needs")
    say(f"  {threshold:+.3f} seconds a lap of free-air advantage for an "
        f"even chance of getting past")
    for value in (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
        say(f"    {value:+.2f} s a lap -> "
            f"{passing.episode_chance(fit, value, gap=mean_gap):5.1%}")

    # ---- the dose is measured, so how good is the measurement? ----------
    trust = pace.reliability(clean, traffic, age_table, arrays["episodes"])
    say()
    say("the pace advantage is measured rather than observed, so the slope "
        "fitted to it is pulled toward zero. Fitting each car's line twice, "
        "on its odd clear laps and on its even ones, says by how much")
    say(f"  the two halves correlate {trust['correlation']:.3f}")
    say(f"  noise {trust['noise sd']:.3f} s a lap against signal "
        f"{trust['signal sd']:.3f}")
    say(f"  reliability {trust['reliability']:.3f}, so the true slope is "
        f"about {fit['per second per lap of advantage'] / trust['reliability']:.2f} "
        f"and the threshold about "
        f"{threshold * trust['reliability']:+.3f} s a lap")

    # ---- per lap, which is a different question --------------------------
    hazard = passing.hazard(arrays, ())
    say()
    say(f"the chance of getting past on any one lap of following "
        f"(n={hazard.n}, {hazard.groups} clusters, McFadden "
        f"{hazard.fit_statistic:.3f})")
    for name in hazard.names:
        low, high = hazard.interval(name)
        say(f"  {name:34s} {hazard[name]:+8.4f} ({low:+.4f} to {high:+.4f})"
            f"  p {hazard.p_value(name):.4f}")
    say("  the time term is negative: a car that has not got past in the "
        "first couple of laps is less likely to manage it on the next one, "
        "not more. Attempts do not accumulate")

    timing = passing.laps_to_pass(arrays)
    say()
    say("how long a pass takes, against how long pure pace says it should")
    say(f"  pace alone predicts {timing['predicted']:.2f} laps, the median "
        f"pass takes {timing['observed']:.2f}")
    say(f"  a ratio of {timing['ratio']:.2f}, over {timing['n']} passes, and "
        f"that understates it: {timing['still behind past the predicted lap']}"
        f" more were still behind past the lap pace said they would be by, "
        f"and never got past at all")

    # ---- does the circuit matter? ----------------------------------------
    circuits = passing.circuit_terms(rows)
    circuit_fit = passing.episode_fit(rows, circuits)
    table = passing.by_circuit(circuit_fit, circuits, rows, gap=mean_gap)
    say()
    say(f"the same threshold per circuit, for the {len(circuits)} circuits "
        f"with enough episodes to carry a term")
    for row in table:
        if not row["named"]:
            continue
        span = ("" if row["low"] is None
                else f" ({row['low']:+.2f} to {row['high']:+.2f})")
        say(f"  {row['circuit']:16s} {row['threshold']:+.2f}{span:26s} "
            f"n={row['episodes']:3d}  got past {row['passes']:3d}")

    joint = linear.joint_test(
        circuit_fit, tuple(f"circuit: {one}" for one in circuits))
    shuffle = passing.permute_circuits(rows, circuits, draws=DRAWS,
                                       gap=mean_gap)
    say()
    say("that column looks like a ranking. It is not one")
    say(f"  jointly, the {joint['terms']} circuit terms give F "
        f"{joint['F']:.3f}, p {joint['p']:.3f}")
    say(f"  the fitted thresholds span {shuffle['observed']:.3f} s a lap; "
        f"shuffling the circuit labels across episodes gives a median span "
        f"of {shuffle['median under shuffling']:.3f}, p {shuffle['p']:.3f}")
    say("  so one season cannot resolve circuit differences at all, and "
        "the apparent ordering is what noise produces on its own")

    # ---- output ------------------------------------------------------------
    figures = charts.draw(
        OUT, agreement=agreement, census=census, bands=bands, fit=fit,
        rows=rows, arrays=arrays, hazard=hazard, table=table,
        air=air, trust=trust, timing=timing, shuffle=shuffle,
        mean_gap=mean_gap, threshold=threshold, circuits=circuits)
    path = report.build(
        agreement=agreement, census=census, pace_census=pace_census,
        bands=bands, fit=fit, hazard=hazard, table=table, air=air,
        trust=trust, timing=timing, shuffle=shuffle, joint=joint,
        rows=rows, field=field, mean_gap=mean_gap, threshold=threshold,
        episodes=episodes, seasons=SEASONS, figures=figures,
        out_path=OUT / "report.html")

    say()
    say(f"report  {path}")
    for name, figure in figures.items():
        say(f"figure  {figure}")
    (OUT / "_run.txt").write_text("\n".join(_log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
