"""Tests that could fail.

Three kinds, and none of them compares against a stored output. The first
pins algebra against a second way of computing the same thing. The second
plants a known truth in a synthetic race and checks the pipeline recovers
it. The third pins the judgement calls that cost real work to get right,
so that undoing one breaks a test with the reason written in it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import follow                                              # noqa: E402
import gaps                                                # noqa: E402
import linear                                              # noqa: E402
import pace                                                # noqa: E402
import passing                                             # noqa: E402
import stints                                              # noqa: E402
from data import Lap, Result, Stop                         # noqa: E402

BASE = 90.0


def _race(*, cars: int = 8, laps: int = 30, pace_step: float = 0.15,
          wear: float = 0.03, stop_lap: int = 15, offset: float = 0.0,
          noise: float = 0.0, reversed_grid: float = 0.0,
          second_stop: int = 0, stagger: int = 0, round_number: int = 1,
          seed: int = 0) -> tuple[list[Lap], list[Stop], list[Result]]:
    """A race where every car's pace is known exactly.

    Car 0 is quickest and car `n` is `n * pace_step` a lap slower, so the
    free-air advantage of car i over car j is a number the test knows
    before anything is fitted. Every car stops on the same lap, which
    keeps tyre age level across the field and out of the comparison.

    `reversed_grid` starts the quickest car at the back, each position
    worth that many seconds. A field whose starting order matches its
    pace order never races: the quick cars lead from lap one and nobody
    ever follows anybody. Reversing it makes the quick cars catch the
    slow ones, so the race produces both laps spent in traffic and laps
    spent in clear air, which is what the pace fit needs to exist at all.

    `second_stop` and `stagger` exist for the same kind of reason. With
    one stop on the same lap for every car, tyre age is race lap minus a
    constant, the two columns of the pace fit are collinear, and the fit
    refuses - correctly. Real races have two or three stops spread over
    a dozen laps, which is what separates them.
    """
    rng = np.random.default_rng(seed)
    laps_out: list[Lap] = []
    stops: list[Stop] = []
    results: list[Result] = []
    for car in range(cars):
        name = f"D{car:02d}"
        own = BASE + car * pace_step
        shift = (car % stagger) if stagger else 0
        car_stops = [stop_lap + shift]
        if second_stop:
            car_stops.append(second_stop + shift)
        car_stops = [one for one in car_stops if one < laps]
        for lap in range(1, laps + 1):
            age = lap
            for one in car_stops:
                if lap > one:
                    age = lap - one
            seconds = own + wear * age
            if lap in car_stops:
                seconds += 3.0
            if lap - 1 in car_stops:
                seconds += 20.0
            if lap == 1:
                seconds += offset * car
                seconds += reversed_grid * (cars - 1 - car)
            if noise:
                seconds += float(rng.normal(0, noise))
            laps_out.append(Lap(season=2024, round=round_number,
                                race="Test", circuit="testring",
                                driver=name, lap=lap,
                                seconds=seconds, position=car + 1))
        for number, one in enumerate(car_stops, start=1):
            stops.append(Stop(season=2024, round=round_number, driver=name,
                              lap=one, stop=number, duration=22.0))
        results.append(Result(season=2024, round=round_number, race="Test",
                              circuit="testring", driver=name, team="T",
                              grid=car + 1, laps=laps, status="Finished",
                              position_text=str(car + 1)))
    return laps_out, stops, results


# ---------------------------------------------------------------- algebra

def test_elapsed_time_is_the_running_sum_of_lap_times():
    laps, _, _ = _race(cars=2, laps=5)
    times = gaps.elapsed(laps)
    car = [row.seconds for row in laps if row.driver == "D00"]
    assert times[(2024, 1, "D00")][3] == pytest.approx(sum(car[:3]))


def test_the_gap_is_the_difference_of_two_running_sums():
    laps, _, _ = _race(cars=2, laps=6, pace_step=0.5)
    times = gaps.elapsed(laps)
    first, second = (2024, 1, "D00"), (2024, 1, "D01")
    # Six laps half a second apart is three seconds, and lap one carries
    # no extra because the synthetic race starts both cars level.
    assert gaps.gap(times, first, second, 6) == pytest.approx(3.0)


def test_order_from_lap_times_is_the_order_the_cars_finished_the_lap():
    laps, _, _ = _race(cars=5, laps=4, pace_step=0.4)
    times = gaps.elapsed(laps)
    assert [key[2] for key in follow.order(times, (2024, 1), 4)] == \
        ["D00", "D01", "D02", "D03", "D04"]


def test_consecutive_laps_group_into_one_span_and_a_break_splits_them():
    assert follow._runs([3, 4, 5, 9, 10]) == [(3, 5), (9, 10)]
    assert follow._runs([7]) == [(7, 7)]


# ------------------------------------------------------- planted recovery

def test_the_free_air_fit_recovers_each_car_s_planted_pace():
    laps, stops, results = _race(cars=6, laps=40, pace_step=0.2, wear=0.04)
    times = gaps.elapsed(laps)
    traffic = follow.in_traffic(laps, times)
    clean, _ = stints.build(laps, stops, results)
    lines, _ = pace.free_air(clean, traffic)
    quick = lines[(2024, 1, "D00")]
    slow = lines[(2024, 1, "D05")]
    # Five steps of 0.2 is a second a lap, read at a lap and age both
    # lines cover.
    assert slow.at(25, 10) - quick.at(25, 10) == pytest.approx(1.0, abs=1e-6)


def test_the_fit_recovers_the_planted_tyre_wear_too():
    laps, stops, results = _race(cars=4, laps=40, wear=0.05)
    times = gaps.elapsed(laps)
    traffic = follow.in_traffic(laps, times)
    clean, _ = stints.build(laps, stops, results)
    lines, _ = pace.free_air(clean, traffic)
    assert lines[(2024, 1, "D00")].beta[1] == pytest.approx(0.05, abs=1e-6)


def test_the_logit_recovers_planted_coefficients():
    rng = np.random.default_rng(11)
    n = 6000
    dose = rng.normal(size=n)
    chance = 1 / (1 + np.exp(-(-0.4 + 1.5 * dose)))
    outcome = (rng.random(n) < chance).astype(float)
    matrix = np.column_stack([np.ones(n), dose])
    fit = linear.logit(outcome, matrix, ("base", "dose"), np.arange(n))
    assert fit["base"] == pytest.approx(-0.4, abs=0.12)
    assert fit["dose"] == pytest.approx(1.5, abs=0.12)


def test_ols_matches_the_normal_equations():
    rng = np.random.default_rng(3)
    matrix = np.column_stack([np.ones(200), rng.normal(size=200)])
    y = matrix @ np.array([2.0, -0.7]) + rng.normal(0, 0.3, 200)
    fit = linear.ols(y, matrix, ("a", "b"), np.arange(200))
    direct = np.linalg.solve(matrix.T @ matrix, matrix.T @ y)
    assert fit.beta == pytest.approx(direct)


# ------------------------------------------------------- what it refuses

def test_a_rank_deficient_design_is_refused_rather_than_fitted():
    column = np.arange(50.0)
    matrix = np.column_stack([np.ones(50), column, 2 * column])
    with pytest.raises(ValueError, match="rank deficient"):
        linear.ols(column, matrix, ("a", "b", "c"), np.arange(50))


def test_a_logit_on_a_constant_outcome_is_refused():
    matrix = np.column_stack([np.ones(40), np.arange(40.0)])
    with pytest.raises(ValueError, match="never varies"):
        linear.logit(np.zeros(40), matrix, ("a", "b"), np.arange(40))


def test_perfect_separation_is_refused_rather_than_reported():
    """A column that predicts the outcome exactly has no finite maximum.

    The coefficients run away while the fit reports a flawless result,
    which is the most dangerous way for a logit to fail.
    """
    dose = np.concatenate([np.full(40, -1.0), np.full(40, 1.0)])
    outcome = (dose > 0).astype(float)
    matrix = np.column_stack([np.ones(80), dose])
    with pytest.raises(ValueError, match="separat"):
        linear.logit(outcome, matrix, ("a", "dose"), np.arange(80))


def test_a_pace_line_is_not_read_outside_the_laps_it_was_fitted_on():
    """The single change that most improved this project's estimate.

    Reading a line at a lap it never saw turned an arbitrary split
    between two collinear coefficients into a pace advantage of twenty
    five seconds a lap, and a handful of such rows flattened the central
    slope by a factor of five.
    """
    line = pace.Line(beta=np.array([90.0, 0.03, 0.01]), lap_lo=10,
                     lap_hi=30, age_lo=1, age_hi=20, n=20)
    lines = {(2024, 1, "D00"): line}
    assert np.isfinite(pace.predict(lines, (2024, 1, "D00"), 20, 10))
    assert not np.isfinite(pace.predict(lines, (2024, 1, "D00"), 45, 10))
    assert not np.isfinite(pace.predict(lines, (2024, 1, "D00"), 20, 40))


def test_an_implausible_pace_advantage_is_refused_rather_than_clipped():
    """Two adjacent cars are not three seconds a lap apart in clear air."""
    wide = pace.Line(beta=np.array([120.0, 0.0, 0.0]), lap_lo=1, lap_hi=50,
                     age_lo=1, age_hi=50, n=40)
    narrow = pace.Line(beta=np.array([90.0, 0.0, 0.0]), lap_lo=1, lap_hi=50,
                       age_lo=1, age_hi=50, n=40)
    lines = {(2024, 1, "A"): wide, (2024, 1, "B"): narrow}
    table = {(2024, 1, "A"): {10: 5}, (2024, 1, "B"): {10: 5}}
    value = pace.advantage(lines, table, (2024, 1, "A"), (2024, 1, "B"), 10)
    assert not np.isfinite(value)


def test_a_line_fitted_inside_one_stint_is_refused_as_not_separable():
    """Inside a stint, tyre age and race lap advance together.

    The two coefficients are then not separately identified and the split
    between them is arbitrary, which only shows up on extrapolation.
    """
    laps, stops, results = _race(cars=3, laps=20, pace_step=1.5,
                                 stop_lap=99)
    times = gaps.elapsed(laps)
    traffic = follow.in_traffic(laps, times)
    clean, _ = stints.build(laps, stops, results)
    _, census = pace.free_air(clean, traffic)
    assert census["refused: age and lap not separable"] == 3
    assert census["fitted car-races"] == 0


# ---------------------------------------------------- episode detection

def test_a_car_that_pits_is_not_recorded_as_having_been_passed():
    """A car dropping to the pit lane loses places without being passed.

    Scoring that as an overtake would credit every pit stop in the season
    to the car behind it.
    """
    laps, stops, results = _race(cars=4, laps=30, pace_step=0.02,
                                 stop_lap=15)
    episodes, census = follow.build(laps, stops, results)
    assert census["dropped: a pit lap"] > 0
    for episode in episodes:
        assert not (episode.start <= 15 <= episode.end)
        assert not (episode.start <= 16 <= episode.end)


def test_neutralised_laps_never_appear_in_an_episode():
    """Overtaking is forbidden behind a safety car."""
    laps, stops, results = _race(cars=6, laps=30)
    slowed = [
        Lap(season=row.season, round=row.round, race=row.race,
            circuit=row.circuit, driver=row.driver, lap=row.lap,
            seconds=row.seconds * 1.6 if 20 <= row.lap <= 22 else row.seconds,
            position=row.position)
        for row in laps]
    neutral = stints.neutralised_laps(slowed)
    assert {20, 21, 22} <= neutral[(2024, 1)]
    episodes, _ = follow.build(slowed, stops, results)
    for episode in episodes:
        assert not set(range(episode.start, episode.end + 1)) & {20, 21, 22}


def test_an_episode_the_race_ends_during_is_dropped_not_scored_as_a_failure():
    laps, stops, results = _race(cars=4, laps=20, pace_step=0.05)
    _, census = follow.build(laps, stops, results)
    assert census["dropped: the race ended first"] > 0


def test_only_adjacent_cars_are_paired():
    """A car two places back is stuck behind the car in front of it."""
    laps, _, _ = _race(cars=4, laps=8, pace_step=0.1)
    times = gaps.elapsed(laps)
    traffic = follow.in_traffic(laps, times)
    for behind, table in traffic.items():
        for lap, ahead in table.items():
            running = [key[2] for key in follow.order(times, (2024, 1), lap)]
            assert running.index(behind[2]) - running.index(ahead[2]) == 1


def test_a_quicker_car_behind_is_recorded_as_getting_past():
    laps, stops, results = _race(cars=2, laps=25, pace_step=-0.5,
                                 offset=1.2, stop_lap=99, wear=0.0)
    episodes, _ = follow.build(laps, stops, results)
    assert episodes and any(episode.passed for episode in episodes)


# ------------------------------------------------------------- modelling

def test_the_opening_gap_is_not_a_mediator_of_the_pace_advantage():
    """Checked rather than assumed, and the first guess was wrong.

    Controlling for the gap looked like it was blocking the pace channel,
    because the fitted slope was far too flat. It was not: the two are
    essentially uncorrelated, and the flatness came from extrapolated
    pace lines instead.
    """
    rng = np.random.default_rng(5)
    n = 900
    dose = rng.normal(size=n)
    gap = rng.uniform(0.1, 2.0, n)
    chance = 1 / (1 + np.exp(-(0.2 + 1.4 * dose - 0.6 * gap)))
    outcome = (rng.random(n) < chance).astype(float)
    rows = {"passed": outcome, "advantage": dose, "gap": gap,
            "circuit": np.full(n, "x"), "group": np.arange(n)}
    with_gap = passing.episode_fit(rows)
    matrix = np.column_stack([np.ones(n), dose])
    without = linear.logit(outcome, matrix,
                           ("base", "per second per lap of advantage"),
                           np.arange(n))
    assert with_gap["per second per lap of advantage"] == pytest.approx(
        without["per second per lap of advantage"], abs=0.2)


def test_the_threshold_inverts_the_fitted_curve():
    rng = np.random.default_rng(7)
    n = 1200
    dose = rng.normal(0.4, 1.0, n)
    gap = rng.uniform(0.2, 2.0, n)
    chance = 1 / (1 + np.exp(-(-0.3 + 1.8 * dose)))
    outcome = (rng.random(n) < chance).astype(float)
    rows = {"passed": outcome, "advantage": dose, "gap": gap,
            "circuit": np.full(n, "x"), "group": np.arange(n)}
    fit = passing.episode_fit(rows)
    point = passing.episode_threshold(fit, gap=1.0)
    assert passing.episode_chance(fit, point, gap=1.0) == pytest.approx(0.5,
                                                                        abs=1e-6)


def test_a_threshold_outside_the_search_range_is_none_not_an_endpoint():
    """A circuit nobody can pass at has no finite answer, and says so."""
    n = 400
    rng = np.random.default_rng(23)
    rows = {"passed": np.zeros(n), "advantage": rng.normal(size=n),
            "gap": rng.uniform(0.2, 2.0, n), "circuit": np.full(n, "x"),
            "group": np.arange(n)}
    # A handful of passes scattered across the whole range of advantage,
    # so the refusal is about the threshold lying outside the search
    # range rather than about a column separating the outcome.
    rows["passed"][rng.choice(n, 8, replace=False)] = 1.0
    fit = passing.episode_fit(rows)
    assert passing.episode_threshold(fit, gap=1.0) is None


def test_episode_length_is_never_a_covariate():
    """It is an outcome: a pass on lap two makes the episode two long."""
    rows = {"passed": np.array([0.0, 1.0]), "advantage": np.array([0.0, 1.0]),
            "gap": np.array([1.0, 1.5]), "laps": np.array([1.0, 2.0]),
            "circuit": np.array(["x", "x"]), "group": np.array([0, 1])}
    first, names = passing.episode_design(rows, ())
    longer = rows | {"laps": np.array([9.0, 40.0])}
    second, _ = passing.episode_design(longer, ())
    assert np.array_equal(first, second)
    assert len(names) == first.shape[1]


def test_the_hazard_and_the_episode_model_agree_on_the_direction():
    laps, stops, results = _race(cars=10, laps=55, pace_step=0.10,
                                 reversed_grid=1.4, second_stop=36,
                                 stagger=5, noise=0.25, seed=2)
    times = gaps.elapsed(laps)
    traffic = follow.in_traffic(laps, times)
    clean, _ = stints.build(laps, stops, results)
    lines, _ = pace.free_air(clean, traffic)
    age_table = pace.ages(stops, laps)
    episodes, _ = follow.build(laps, stops, results)
    arrays = passing.trials(episodes, lines, age_table)
    rows = passing.episodes_of(arrays)
    if rows["passed"].min() == rows["passed"].max():
        pytest.skip("this synthetic race produced no variation to fit")
    assert (passing.episode_fit(rows)["per second per lap of advantage"]
            > 0) == (passing.hazard(arrays, ())
                     ["per second per lap of advantage"] > 0)


def test_collapsing_to_episodes_keeps_every_episode_exactly_once():
    arrays = {"passed": np.array([0.0, 0.0, 1.0, 0.0]),
              "advantage": np.array([0.5, 0.5, 0.5, -0.2]),
              "gap": np.array([1.0, 1.0, 1.0, 1.5]),
              "lap": np.array([1.0, 2.0, 3.0, 1.0]),
              "circuit": np.array(["a", "a", "a", "b"]),
              "group": np.array([0, 0, 0, 1])}
    rows = passing.episodes_of(arrays)
    assert rows["passed"].size == 2
    assert rows["passed"].tolist() == [1.0, 0.0]
    assert rows["laps"].tolist() == [3.0, 1.0]


# -------------------------------------------------------------- checking

def test_the_joint_test_sees_what_single_intervals_miss():
    """Several terms can each look like zero and jointly be far from it."""
    rng = np.random.default_rng(13)
    n = 800
    blocks = np.column_stack([(rng.integers(0, 4, n) == k).astype(float)
                              for k in range(3)])
    y = 1.0 + blocks @ np.array([0.4, 0.4, 0.4]) + rng.normal(0, 1.0, n)
    matrix = np.column_stack([np.ones(n), blocks])
    fit = linear.ols(y, matrix, ("base", "a", "b", "c"), np.arange(n))
    assert linear.joint_test(fit, ("a", "b", "c"))["p"] < 0.05


def test_shuffling_the_labels_destroys_a_real_effect_and_keeps_the_rest():
    rng = np.random.default_rng(17)
    n = 600
    circuit = rng.choice(["a", "b", "c"], n)
    dose = rng.normal(size=n)
    gap = rng.uniform(0.2, 2.0, n)
    # A real pace effect and no circuit effect at all, so the thresholds
    # are finite and any spread between them has to be noise.
    chance = 1 / (1 + np.exp(-(0.1 + 1.6 * dose - 0.5 * gap)))
    rows = {"passed": (rng.random(n) < chance).astype(float),
            "advantage": dose, "gap": gap,
            "circuit": circuit, "group": np.arange(n)}
    out = passing.permute_circuits(rows, ("a", "b"), draws=40, gap=1.0)
    # The labels carry nothing, so the real spread is an ordinary draw
    # from the shuffled ones and the p-value must not be small.
    assert out["p"] > 0.05


def test_split_half_reliability_falls_when_the_laps_are_noisier():
    """More noise in the lap times means less signal in the measured dose."""
    # Both levels have to be noisy enough that cars actually swap places;
    # a perfectly orderly synthetic race has almost no episodes to split.
    scores = []
    for noise in (0.3, 0.8):
        laps: list[Lap] = []
        stops: list[Stop] = []
        results: list[Result] = []
        # Several races pooled, as the real season is: splitting a single
        # car-race's clear laps in half leaves too few on each side to
        # estimate a variance from.
        for number in range(1, 21):
            one, two, three = _race(cars=12, laps=58, pace_step=0.10,
                                    reversed_grid=1.2, second_stop=36,
                                    stagger=5, noise=noise,
                                    round_number=number, seed=number)
            laps += one
            stops += two
            results += three
        times = gaps.elapsed(laps)
        traffic = follow.in_traffic(laps, times)
        clean, _ = stints.build(laps, stops, results)
        lines, _ = pace.free_air(clean, traffic)
        age_table = pace.ages(stops, laps)
        episodes, _ = follow.build(laps, stops, results)
        arrays = passing.trials(episodes, lines, age_table)
        trust = pace.reliability(clean, traffic, age_table,
                                 arrays["episodes"])
        assert trust["n"] > 100
        scores.append(trust["reliability"])
    assert scores[0] > scores[1]


def test_being_near_a_car_you_are_not_quicker_than_is_the_control():
    """The number that makes the dirty-air measurement mean something.

    Without it, a penalty measured on every lap within two seconds could
    just as well be cars that were slow for their own reasons.
    """
    laps, stops, results = _race(cars=10, laps=55, pace_step=0.10,
                                 reversed_grid=1.4, second_stop=36,
                                 stagger=5, noise=0.2, seed=6)
    times = gaps.elapsed(laps)
    traffic = follow.in_traffic(laps, times)
    clean, _ = stints.build(laps, stops, results)
    lines, _ = pace.free_air(clean, traffic)
    age_table = pace.ages(stops, laps)
    held = pace.held_up_laps(traffic, lines, age_table)
    air = pace.dirty_air(clean, lines, traffic, held)
    assert "penalty when merely near" in air
    assert "penalty when held up" in air


def test_laps_to_pass_ignores_cars_that_are_barely_quicker():
    """The prediction divides by the advantage, so near zero it explodes."""
    arrays = {"passed": np.array([1.0, 1.0]),
              "advantage": np.array([0.01, 1.0]),
              "gap": np.array([1.0, 1.0]), "lap": np.array([1.0, 1.0]),
              "circuit": np.array(["a", "a"]), "group": np.array([0, 1])}
    assert passing.laps_to_pass(arrays)["n"] == 1


def test_a_circuit_with_no_passes_gets_no_term_of_its_own():
    """Its intercept is not identified, and the fit would separate."""
    rows = {"passed": np.array([0.0, 0.0, 1.0] * 20),
            "advantage": np.zeros(60), "gap": np.ones(60),
            "circuit": np.array(["empty"] * 30 + ["mixed"] * 30),
            "group": np.arange(60)}
    rows["passed"][:30] = 0.0
    assert "empty" not in passing.circuit_terms(rows, minimum=10, passes=3)
