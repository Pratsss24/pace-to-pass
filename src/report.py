"""One self-contained HTML file: charts inlined, no assets, no server."""

from __future__ import annotations

import base64
from pathlib import Path

CSS = """
body{font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1120px;
margin:40px auto;padding:0 20px;color:#16181d}
h1{margin-bottom:4px}h2{margin-top:38px;border-bottom:1px solid #e3e6ea;
padding-bottom:6px}
.sub{color:#5b6472;margin-top:0}
.tiles{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}
.tile{flex:1;min-width:150px;background:#f6f8fa;border:1px solid #e3e6ea;
border-radius:9px;padding:14px 16px}
.tile .v{font-size:25px;font-weight:650;letter-spacing:-.5px}
.tile .k{font-size:11px;text-transform:uppercase;letter-spacing:.07em;
color:#5b6472;margin-top:3px}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}
th,td{border:1px solid #e3e6ea;padding:7px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:#f6f8fa;font-weight:600}
td.good{color:#17803d}td.bad{color:#b4232c}
img{width:100%;margin:14px 0;border:1px solid #e3e6ea;border-radius:7px}
.note{background:#fff8e6;border-left:3px solid #d9a441;padding:11px 15px;
margin:18px 0;font-size:14px}
code{background:#f0f2f5;padding:1px 5px;border-radius:4px;font-size:13px}
"""


def _img(path: Path) -> str:
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{Path(path).stem}">'


def build(*, agreement, census, pace_census, bands, fit, hazard, table, air,
          trust, timing, shuffle, joint, rows, field, mean_gap, threshold,
          episodes, seasons, figures, out_path: Path) -> Path:
    tiles = [
        ("pace needed to get past", f"{threshold:+.2f} s a lap"),
        ("episodes measured", f"{rows['passed'].size:,}"),
        ("cost of being held up", f"{air['penalty when held up']:+.2f} s"),
        ("circuit differences", f"p {shuffle['p']:.2f}"),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div></div>'
        for k, v in tiles)

    census_html = "".join(
        f"<tr><td>{key}</td><td>{value:,}</td></tr>"
        for key, value in sorted(census.items(), key=lambda item: -item[1]))
    pace_html = "".join(
        f"<tr><td>{key}</td><td>{value:,}</td></tr>"
        for key, value in sorted(pace_census.items(),
                                 key=lambda item: -item[1]))

    bands_html = "".join(
        f"<tr><td>{band['low']:+.1f} to {band['high']:+.1f}</td>"
        f"<td>{band['n']}</td>"
        f'<td class="{"good" if band["share"] > 0.5 else ""}">'
        f"{band['share']:.1%} &plusmn; {1.96 * band['error']:.1%}</td></tr>"
        for band in bands)

    fit_html = "".join(
        f"<tr><td>{name}</td><td>{fit[name]:+.4f}</td>"
        f"<td>{fit.interval(name)[0]:+.4f} to "
        f"{fit.interval(name)[1]:+.4f}</td>"
        f"<td>{fit.p_value(name):.4f}</td></tr>" for name in fit.names)

    hazard_html = "".join(
        f"<tr><td>{name}</td><td>{hazard[name]:+.4f}</td>"
        f"<td>{hazard.interval(name)[0]:+.4f} to "
        f"{hazard.interval(name)[1]:+.4f}</td>"
        f"<td>{hazard.p_value(name):.4f}</td></tr>" for name in hazard.names)

    def circuit_row(row) -> str:
        if row["low"] is None or row["high"] is None:
            span = "&mdash;"
        else:
            span = f'{row["low"]:+.2f} to {row["high"]:+.2f}'
        return (f'<tr><td>{row["circuit"]}</td>'
                f'<td>{row["threshold"]:+.2f}</td><td>{span}</td>'
                f'<td>{row["episodes"]}</td><td>{row["passes"]}</td></tr>')

    circuit_html = "".join(
        circuit_row(row) for row in table
        if row["named"] and row["threshold"] is not None)

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Pace to pass</title><style>{CSS}</style></head><body>
<h1>Pace to pass</h1>
<p class="sub">How much quicker a car has to be to get past the one in
front, measured from {"-".join(str(s) for s in seasons)} lap times alone
&mdash; and whether the circuit makes any difference.</p>

<div class="tiles">{tiles_html}</div>

<div class="note"><b>The race order is in the lap times.</b> Summing a
car's lap times gives its elapsed race time, so sorting cars by that
total at lap n puts them in the order they crossed the line. Against the
position the feed reports, this agrees on
<b>{agreement['agree']:,} of {agreement['agree'] + agreement['disagree']:,}</b>
laps &mdash; {agreement['share']:.2%}. Nothing here needs a positional
feed, a gap feed or telemetry.</div>

<h2>Which pairs are measured</h2>
{_img(figures['dose'])}
<table><tr><th>outcome</th><th>episodes</th></tr>{census_html}</table>

<h2>How quick each car was</h2>
<p>The dose cannot be read off the laps two cars spend together, because
those laps are the outcome: a car stuck behind another is slowed by being
stuck behind it. So pace is measured apart and tested together &mdash;
each car-race gets a line fitted to the laps it ran in clear air, and the
advantage is the difference of two lines, each read at that car's own
tyre age. The fitted field spans {field['median spread']:.2f} s a lap at
the median race, which is what a Formula 1 grid covers.</p>
<table><tr><th>outcome</th><th>laps or car-races</th></tr>{pace_html}</table>

<h2>What being held up costs</h2>
{_img(figures['air'])}
<p>The second bar is the control. Being close to another car costs
nothing in itself &mdash; <b>{air['penalty when merely near']:+.3f} s</b> a
lap for cars that were not quicker anyway. The cost appears only where
the car behind had the pace to go faster and could not:
<b>{air['penalty when held up']:+.3f} s</b> a lap. Pooling the two, which
is the obvious thing to do, dilutes it to almost nothing.</p>

<h2>How much quicker you have to be</h2>
<table><tr><th>free-air advantage</th><th>episodes</th>
<th>got past</th></tr>{bands_html}</table>
<table><tr><th>term</th><th>estimate</th><th>95% interval</th><th>p</th></tr>
{fit_html}</table>
<p>At the mean opening gap of {mean_gap:.2f} s, the fitted curve crosses
even odds at <b>{threshold:+.2f} seconds a lap</b>. The intercept is
{fit['base']:+.3f}: two cars of identical pace running nose to tail are a
coin toss, which is what it ought to be and was not imposed.</p>

<h2>The dose is measured, not observed</h2>
{_img(figures['checks'])}
<p>Fitting each car's line twice &mdash; once on its odd clear laps, once
on its even ones &mdash; gives two independent measurements of the same
quantity. They correlate {trust['correlation']:.3f}, leaving noise of
{trust['noise sd']:.3f} s a lap against signal of
{trust['signal sd']:.3f}. Reliability is {trust['reliability']:.3f}, so
the fitted slope is pulled toward zero by about that factor: the true
slope is nearer
{fit['per second per lap of advantage'] / trust['reliability']:.2f} and
the threshold nearer {threshold * trust['reliability']:+.2f} s a lap.</p>

<h2>Attempts do not accumulate</h2>
{_img(figures['hazard'])}
<table><tr><th>term</th><th>estimate</th><th>95% interval</th><th>p</th></tr>
{hazard_html}</table>
<p>The time term is negative. A car that has not got past in the first
couple of laps is <i>less</i> likely to manage it on the next one, not
more. And a pass takes {timing['ratio']:.2f} times as long as pure pace
says it should &mdash; {timing['observed']:.2f} laps against
{timing['predicted']:.2f} &mdash; which understates the resistance,
because {timing['still behind past the predicted lap']} more cars were
still behind past the lap pace said they would be by and never got past
at all.</p>

<h2>Does the circuit matter? Not measurably</h2>
{_img(figures['circuits'])}
<table><tr><th>circuit</th><th>threshold</th><th>95% interval</th>
<th>episodes</th><th>got past</th></tr>{circuit_html}</table>
<div class="note"><b>That column is not a ranking.</b> Jointly, the
{joint['terms']} circuit terms give F {joint['F']:.3f}, p
{joint['p']:.3f}. The fitted thresholds span
{shuffle['observed']:.3f} s a lap; shuffling the circuit labels across
episodes &mdash; which destroys any real circuit effect and keeps
everything else &mdash; gives a median span of
{shuffle['median under shuffling']:.3f}, and beats the real spread
{shuffle['p']:.0%} of the time. One season cannot resolve circuit
differences, and the apparent ordering is what noise produces on its
own.</div>

<h2>What the guards were worth</h2>
{_img(figures['guards'])}
<p>A per-car line read outside the laps and tyre ages it was fitted on
returned pace advantages of twenty-five seconds a lap. Two per cent of
the rows were enough to flatten the central slope by a factor of five.
Refusing implausible advantages recovers most of it; refusing
extrapolation altogether recovers the rest, and the estimate then stops
moving as the rule is tightened further.</p>

<p class="sub">Generated by <code>run.py</code>. Lap times and pit stops
from the Jolpica mirror of the Ergast schema. The cleaning rules are
carried over from the stint-degradation, race-strategy-mc and
undercut-effect projects.</p>
</body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
