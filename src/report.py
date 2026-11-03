"""Build the report as a printable PDF.

`paper.py` owns the page furniture; everything here is this project's own
argument, in the order the argument runs.
"""

from __future__ import annotations

from pathlib import Path

from paper import Paper

SOURCE = ("Lap times and pit stops from the Jolpica mirror of the Ergast "
          "schema. The cleaning rules are carried over from the "
          "stint-degradation, race-strategy-mc and undercut-effect "
          "projects.")


def _counts(counts: dict, head: str) -> tuple[list[str], list[list[str]]]:
    rows = [[key, f"{value:,}"] for key, value
            in sorted(counts.items(), key=lambda item: -item[1])]
    return (["outcome", head], rows)


def _bands(bands) -> tuple[list[str], list[list[str]]]:
    rows = [[f"{one['low']:+.1f} to {one['high']:+.1f}", str(one["n"]),
             f"{one['share']:.1%} ± {1.96 * one['error']:.1%}"]
            for one in bands]
    return (["free-air advantage", "episodes", "got past"], rows)


def _terms(fit) -> tuple[list[str], list[list[str]]]:
    rows = []
    for name in fit.names:
        low, high = fit.interval(name)
        rows.append([name, f"{fit[name]:+.4f}",
                     f"{low:+.4f} to {high:+.4f}",
                     f"{fit.p_value(name):.4f}"])
    return (["term", "estimate", "95% interval", "p"], rows)


def _circuits(table) -> tuple[list[str], list[list[str]]]:
    rows = []
    for one in table:
        if not one["named"] or one["threshold"] is None:
            continue
        if one["low"] is None or one["high"] is None:
            span = "—"
        else:
            span = f"{one['low']:+.2f} to {one['high']:+.2f}"
        rows.append([one["circuit"], f"{one['threshold']:+.2f}", span,
                     str(one["episodes"]), str(one["passes"])])
    return (["circuit", "threshold", "95% interval", "episodes",
             "got past"], rows)


def build(*, agreement, census, pace_census, bands, fit, hazard, table, air,
          trust, timing, shuffle, joint, rows, field, mean_gap, threshold,
          episodes, seasons, figures, out_path: Path) -> Path:
    checked = agreement["agree"] + agreement["disagree"]
    out = Path(out_path)
    out = out.with_name("report.pdf") if out.suffix != ".pdf" else out
    out.parent.mkdir(parents=True, exist_ok=True)

    with Paper(out, "Pace to pass",
               f"How much quicker a car has to be to get past the one in "
               f"front, measured from "
               f"{'-'.join(str(one) for one in seasons)} lap times alone "
               f"— and whether the circuit makes any difference.") \
            as paper:

        paper.cover(
            headline=f"It takes {threshold:+.2f} seconds a lap to get past",
            small=f"At the mean opening gap of {mean_gap:.2f} s, over "
                  f"{rows['passed'].size:,} episodes. Being held up costs "
                  f"{air['penalty when held up']:+.2f} s a lap, and the "
                  f"circuit makes no measurable difference "
                  f"(p {shuffle['p']:.2f}).",
            tiles=[("pace needed to get past", f"{threshold:+.2f} s"),
                   ("episodes measured", f"{rows['passed'].size:,}"),
                   ("cost of being held up",
                    f"{air['penalty when held up']:+.2f} s"),
                   ("circuit differences", f"p {shuffle['p']:.2f}")],
            blurb=f"The race order is in the lap times. Summing a car's lap "
                  f"times gives its elapsed race time, so sorting cars by "
                  f"that total at lap n puts them in the order they crossed "
                  f"the line. Against the position the feed reports, this "
                  f"agrees on {agreement['agree']:,} of {checked:,} laps "
                  f"— {agreement['share']:.2%}. Nothing here needs a "
                  f"positional feed, a gap feed or telemetry.",
            figure_path=figures["dose"])

        paper.section("Which pairs are measured",
                      table=_counts(census, "episodes"))

        paper.section(
            "How quick each car was",
            body=f"The dose cannot be read off the laps two cars spend "
                 f"together, because those laps are the outcome: a car "
                 f"stuck behind another is slowed by being stuck behind it. "
                 f"So pace is measured apart and tested together — "
                 f"each car-race gets a line fitted to the laps it ran in "
                 f"clear air, and the advantage is the difference of two "
                 f"lines, each read at that car's own tyre age. The fitted "
                 f"field spans {field['median spread']:.2f} s a lap at the "
                 f"median race, which is what a Formula 1 grid covers.",
            table=_counts(pace_census, "laps or car-races"))

        paper.section(
            "What being held up costs",
            warn=f"The second bar is the control. Being close to another "
                 f"car costs nothing in itself — "
                 f"{air['penalty when merely near']:+.3f} s a lap for cars "
                 f"that were not quicker anyway. The cost appears only "
                 f"where the car behind had the pace to go faster and could "
                 f"not: {air['penalty when held up']:+.3f} s a lap. Pooling "
                 f"the two, which is the obvious thing to do, dilutes it to "
                 f"almost nothing.",
            figure_path=figures["air"])

        paper.section(
            "How much quicker you have to be",
            body=f"At the mean opening gap of {mean_gap:.2f} s, the fitted "
                 f"curve crosses even odds at {threshold:+.2f} seconds a "
                 f"lap. The intercept is {fit['base']:+.3f}: two cars of "
                 f"identical pace running nose to tail are a coin toss, "
                 f"which is what it ought to be and was not imposed.",
            table=_bands(bands))

        paper.section(
            "The fitted curve",
            note="One logistic fit over every episode, with the opening gap "
                 "as a control.",
            table=_terms(fit))

        paper.section(
            "The dose is measured, not observed",
            warn=f"Fitting each car's line twice — once on its odd "
                 f"clear laps, once on its even ones — gives two "
                 f"independent measurements of the same quantity. They "
                 f"correlate {trust['correlation']:.3f}, leaving noise of "
                 f"{trust['noise sd']:.3f} s a lap against signal of "
                 f"{trust['signal sd']:.3f}. Reliability is "
                 f"{trust['reliability']:.3f}, so the fitted slope is "
                 f"pulled toward zero by about that factor: the true slope "
                 f"is nearer "
                 f"{fit['per second per lap of advantage'] / trust['reliability']:.2f}"
                 f" and the threshold nearer "
                 f"{threshold * trust['reliability']:+.2f} s a lap.",
            figure_path=figures["checks"])

        paper.section(
            "Attempts do not accumulate",
            body=f"The time term is negative. A car that has not got past "
                 f"in the first couple of laps is less likely to manage it "
                 f"on the next one, not more. And a pass takes "
                 f"{timing['ratio']:.2f} times as long as pure pace says it "
                 f"should — {timing['observed']:.2f} laps against "
                 f"{timing['predicted']:.2f} — which understates the "
                 f"resistance, because "
                 f"{timing['still behind past the predicted lap']} more "
                 f"cars were still behind past the lap pace said they would "
                 f"be by and never got past at all.",
            figure_path=figures["hazard"], table=_terms(hazard))

        paper.section(
            "Does the circuit matter? Not measurably",
            warn=f"That column is not a ranking. Jointly, the "
                 f"{joint['terms']} circuit terms give F {joint['F']:.3f}, "
                 f"p {joint['p']:.3f}. The fitted thresholds span "
                 f"{shuffle['observed']:.3f} s a lap; shuffling the circuit "
                 f"labels across episodes — which destroys any real "
                 f"circuit effect and keeps everything else — gives a "
                 f"median span of {shuffle['median under shuffling']:.3f}, "
                 f"and beats the real spread {shuffle['p']:.0%} of the "
                 f"time. One season cannot resolve circuit differences, and "
                 f"the apparent ordering is what noise produces on its own.",
            figure_path=figures["circuits"], table=_circuits(table))

        paper.section(
            "What the guards were worth",
            body="A per-car line read outside the laps and tyre ages it was "
                 "fitted on returned pace advantages of twenty-five seconds "
                 "a lap. Two per cent of the rows were enough to flatten "
                 "the central slope by a factor of five. Refusing "
                 "implausible advantages recovers most of it; refusing "
                 "extrapolation altogether recovers the rest, and the "
                 "estimate then stops moving as the rule is tightened "
                 "further.",
            figure_path=figures["guards"], after=SOURCE)

    return out
