"""When one car is running close behind another, and how that ends.

An *episode* is a run of consecutive laps on which one car is directly
behind another and within reach of it. It ends when the order flips - a
pass - or when the chasing car drops back out of reach, or when something
interrupts it. Each episode is one observation of the question this
project asks: given that a car is right behind another, does it get past?

**Race order comes out of the lap times.** `gaps.elapsed` gives each car's
elapsed race time at the end of every lap, so sorting cars by their
elapsed time at lap n puts them in the order they crossed the line on lap
n. That is the race order, and the difference between two neighbours is
the gap. No positional feed is needed to produce either.

The feed does carry a reported position, and `order_agreement` compares
the two. It is a check on the instrument rather than an input: if the
order derived from lap times disagreed with the reported one in any
meaningful share of laps, the arithmetic above would be wrong and
everything downstream with it.

**Directly behind.** Only neighbours in that order are paired. A car two
places back is not attempting a pass on the leader of the group; it is
stuck behind the car in front of it, and the pair that matters is the
adjacent one.

**What is excluded, and why.**

- *Lap one* is the start, where positions change for reasons that have
  nothing to do with one car being quicker than another over a lap.
- *Neutralised laps* - overtaking is forbidden behind a safety car, so a
  lap run under one is not a trial of anything.
- *Pit laps*, the in-lap and the out-lap of either car. A car that pits
  from second place drops back without having been passed on track, and
  the car behind inherits the place without having done anything.
- *Episodes that cannot be resolved*: the race ends, a car retires, or a
  pit stop interrupts while the pair is still together. These are dropped
  rather than counted as failures to pass, which would quietly load every
  interruption onto the "did not get past" side.

**Lapped cars.** A car a lap down has an elapsed time a full lap larger,
so it sorts into its true race position and shows a gap of eighty seconds
or more to the car ahead of it. The reach filter removes those pairs. It
is a filter that is load-bearing for a second reason, which is worth
knowing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

import gaps
import stints
from data import Lap, Result, Stop

REACH = 2.0          # seconds: close enough that a pass is in prospect
FIRST_LAP = 2        # lap one is the start, not a lap of racing

Key = tuple[int, int, str]


@dataclass(frozen=True)
class Episode:
    """One run of laps spent directly behind another car."""

    season: int
    round: int
    circuit: str
    ahead: str
    behind: str
    start: int           # first lap of following
    end: int             # last lap of following
    passed: bool
    gap_start: float     # seconds behind at the first lap of the episode

    @property
    def laps(self) -> int:
        return self.end - self.start + 1

    @property
    def race(self) -> tuple[int, int]:
        return (self.season, self.round)

    @property
    def key(self) -> Key:
        """The chasing car-race: the unit the errors are clustered on."""
        return (self.season, self.round, self.behind)

    @property
    def name(self) -> tuple:
        """Identifies the episode itself."""
        return (self.season, self.round, self.ahead, self.behind, self.start)


def pit_laps(stops: list[Stop]) -> dict[Key, set[int]]:
    """The in-lap and out-lap of every stop, per car-race.

    Both, because the pit-lane time lands mostly on the out-lap: a car
    that stops on lap L is slow on lap L and much slower on lap L+1, and
    neither lap says anything about whether it could pass.
    """
    table: dict[Key, set[int]] = defaultdict(set)
    for stop in stops:
        table[(stop.season, stop.round, stop.driver)].update(
            (stop.lap, stop.lap + 1))
    return dict(table)


def usable(laps: list[Lap], stops: list[Stop], neutral: dict
           ) -> dict[Key, set[int]]:
    """Laps on which a car could have raced, per car-race."""
    pits = pit_laps(stops)
    table: dict[Key, set[int]] = defaultdict(set)
    for lap in laps:
        key = (lap.season, lap.round, lap.driver)
        if lap.lap < FIRST_LAP:
            continue
        if lap.lap in neutral.get((lap.season, lap.round), ()):
            continue
        if lap.lap in pits.get(key, ()):
            continue
        table[key].add(lap.lap)
    return dict(table)


def order(times: dict[Key, np.ndarray], race: tuple[int, int],
          lap: int) -> list[Key]:
    """The cars that completed `lap`, in the order they completed it."""
    running = [(value[lap], key) for key, value in times.items()
               if key[:2] == race and lap < value.size
               and np.isfinite(value[lap])]
    running.sort()
    return [key for _, key in running]


def order_agreement(laps: list[Lap], times: dict[Key, np.ndarray]) -> dict:
    """How often the order from lap times matches the reported position.

    A check on the arithmetic, not an input to anything. Lap one is
    excluded for the same reason it is excluded everywhere else.
    """
    reported: dict[tuple[int, int, int], dict[Key, int]] = defaultdict(dict)
    for lap in laps:
        if lap.lap < FIRST_LAP:
            continue
        reported[(lap.season, lap.round, lap.lap)][
            (lap.season, lap.round, lap.driver)] = lap.position

    agree = disagree = 0
    for (season, round_, number), table in reported.items():
        derived = order(times, (season, round_), number)
        if len(derived) != len(table):
            continue
        for place, key in enumerate(derived, start=1):
            if key not in table:
                continue
            if table[key] == place:
                agree += 1
            else:
                disagree += 1
    total = agree + disagree
    return {"agree": agree, "disagree": disagree,
            "share": agree / total if total else float("nan")}


def _runs(numbers: list[int]) -> list[tuple[int, int]]:
    """Consecutive integers grouped into (first, last) spans."""
    spans: list[tuple[int, int]] = []
    for number in sorted(numbers):
        if spans and number == spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], number)
        else:
            spans.append((number, number))
    return spans


def build(laps: list[Lap], stops: list[Stop], results: list[Result],
          *, reach: float = REACH, neutral: dict | None = None
          ) -> tuple[list[Episode], dict[str, int]]:
    """Find every following episode and say how each one ended."""
    if neutral is None:
        neutral = stints.neutralised_laps(laps)
    times = gaps.elapsed(laps)
    open_laps = usable(laps, stops, neutral)
    races = sorted({(lap.season, lap.round) for lap in laps})
    circuits = {(lap.season, lap.round): lap.circuit for lap in laps}

    census = {"pairs within reach": 0, "dropped: a pit lap": 0,
              "dropped: the race ended first": 0,
              "dropped: a car stopped running": 0,
              "kept: ended in a pass": 0, "kept: never got past": 0}

    episodes: list[Episode] = []
    for race in races:
        last_lap = max((value.size - 1 for key, value in times.items()
                        if key[:2] == race), default=0)
        close: dict[tuple[Key, Key], list[int]] = defaultdict(list)
        for lap in range(FIRST_LAP, last_lap + 1):
            running = order(times, race, lap)
            for ahead, behind in zip(running, running[1:]):
                between = times[behind][lap] - times[ahead][lap]
                if not 0 < between <= reach:
                    continue
                census["pairs within reach"] += 1
                if (lap not in open_laps.get(ahead, ())
                        or lap not in open_laps.get(behind, ())):
                    census["dropped: a pit lap"] += 1
                    continue
                close[(ahead, behind)].append(lap)

        for (ahead, behind), following in close.items():
            for start, end in _runs(following):
                after = end + 1
                if after > last_lap:
                    census["dropped: the race ended first"] += 1
                    continue
                first, second = times[ahead], times[behind]
                if (after >= first.size or after >= second.size
                        or not np.isfinite(first[after])
                        or not np.isfinite(second[after])):
                    census["dropped: a car stopped running"] += 1
                    continue
                overtook = bool(second[after] < first[after])
                census["kept: ended in a pass" if overtook
                       else "kept: never got past"] += 1
                episodes.append(Episode(
                    season=race[0], round=race[1], circuit=circuits[race],
                    ahead=ahead[2], behind=behind[2], start=start, end=end,
                    passed=overtook,
                    gap_start=float(second[start] - first[start])))

    return episodes, census


def following_laps(episodes: list[Episode]) -> dict[Key, set[int]]:
    """Every lap each car spent stuck behind somebody, per car-race.

    Used to hold those laps out of the free-air pace fit: a lap run in
    another car's wake is not a measurement of how quick the car is.
    """
    table: dict[Key, set[int]] = defaultdict(set)
    for episode in episodes:
        key = (episode.season, episode.round, episode.behind)
        table[key].update(range(episode.start, episode.end + 1))
    return dict(table)


def in_traffic(laps: list[Lap], times: dict[Key, np.ndarray], *,
               reach: float = REACH) -> dict[Key, dict[int, Key]]:
    """Every lap a car ran within reach of the car directly ahead, and who.

    Broader than the episodes, and deliberately so. An episode has to be
    resolvable to be kept, but a lap run in another car's wake is spoiled
    as a measurement of pace whether or not the episode it belongs to
    could be scored. This is the set held out of the free-air fit, and
    the car ahead is carried along because the pace comparison needs it.
    """
    races = sorted({(lap.season, lap.round) for lap in laps})
    table: dict[Key, dict[int, Key]] = defaultdict(dict)
    for race in races:
        last_lap = max((value.size - 1 for key, value in times.items()
                        if key[:2] == race), default=0)
        for lap in range(FIRST_LAP, last_lap + 1):
            running = order(times, race, lap)
            for ahead, behind in zip(running, running[1:]):
                between = times[behind][lap] - times[ahead][lap]
                if 0 < between <= reach:
                    table[behind][lap] = ahead
    return dict(table)
