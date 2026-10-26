"""
Turns per-chart model scores into decisions, costs and staffing.

Shared by train.py (to pick the operating point) and app.py (to redraw it live
when the sliders move), so both always compute cost the same way.

One case is one chart. Each chart has:
  truth : LONG, SHORT or NONE, from its labelled patterns
  score : confidence of the model's most confident box (0 if it found nothing)
  side  : LONG or SHORT, the direction of that most confident box

Two thresholds turn a score into a decision:
  score <  t_low            -> NONE   (no trade, nobody looks)
  score >= t_high           -> trade the box's side automatically
  t_low <= score < t_high   -> REVIEW (an analyst looks and gets it right)
With t_low == t_high there is no review band and the model decides alone.
"""

from dataclasses import dataclass, replace

import numpy as np

TRUTHS = ("LONG", "SHORT", "NONE")
DECISIONS = ("LONG", "SHORT", "NONE", "REVIEW")
LONG, SHORT, NONE, REVIEW = 0, 1, 2, 3
GRID = np.round(np.linspace(0.0, 1.0, 101), 2)


def normalise(name: str) -> str:
    return " ".join(str(name).lower().replace("_", " ").replace("-", " ").split())


def side_of(class_name: str, class_side: dict) -> int:
    lookup = {normalise(k): v for k, v in class_side.items()}
    key = normalise(class_name)
    if key not in lookup:
        raise KeyError(
            f"Class '{class_name}' has no direction in config.CLASS_SIDE. "
            f"Add it as LONG or SHORT."
        )
    return LONG if lookup[key] == "LONG" else SHORT


def chart_truth(gt_boxes: list, names: list, class_side: dict, rule: str = "latest") -> int:
    """Label a chart LONG, SHORT or NONE from its ground truth boxes.

    gt_boxes rows are [cls, x1, y1, x2, y2] in 0..1 image coordinates.
    """
    if not gt_boxes:
        return NONE
    sides = {side_of(names[int(b[0])], class_side) for b in gt_boxes}
    if len(sides) == 1:
        return sides.pop()
    if rule != "latest":
        raise ValueError(f"Unknown MIXED_CHART_RULE '{rule}'")
    latest = max(gt_boxes, key=lambda b: b[3])  # furthest right edge
    return side_of(names[int(latest[0])], class_side)


def chart_score(pred_boxes: list, names: list, class_side: dict) -> tuple[float, int]:
    """Score and side of a chart from its predicted boxes.

    pred_boxes rows are [cls, conf, x1, y1, x2, y2].
    """
    if not pred_boxes:
        return 0.0, LONG
    top = max(pred_boxes, key=lambda b: b[1])
    return float(top[1]), side_of(names[int(top[0])], class_side)


def decide_one(score: float, side: int, t_low: float, t_high: float) -> str:
    """The decision for a single chart, as a word."""
    if score < t_low:
        return "NONE"
    if score >= max(t_low, t_high):
        return TRUTHS[side]
    return "REVIEW"


@dataclass(frozen=True)
class Economics:
    cost_missed: float
    cost_spurious: float
    cost_wrong_direction: float
    cost_correct_trade: float
    cost_correct_pass: float
    review_cost: float          # euros per chart an analyst checks
    base_rate: float | None     # share of charts with a real pattern, None = use the data's
    charts_per_day: float
    minutes_per_review: float
    review_hours_per_analyst: float
    max_analysts: float
    reviewer_accuracy: float = 1.0  # chance the analyst makes the right call

    @classmethod
    def from_config(cls, cfg, base_rate: float | None = "config") -> "Economics":
        return cls(
            cost_missed=cfg.COST_MISSED_SETUP,
            cost_spurious=cfg.COST_SPURIOUS_TRADE,
            cost_wrong_direction=cfg.COST_WRONG_DIRECTION,
            cost_correct_trade=cfg.COST_CORRECT_TRADE,
            cost_correct_pass=cfg.COST_CORRECT_PASS,
            review_cost=cfg.MINUTES_PER_REVIEW / 60.0 * cfg.ANALYST_COST_PER_HOUR,
            base_rate=cfg.BASE_RATE if base_rate == "config" else base_rate,
            charts_per_day=cfg.CHARTS_PER_DAY,
            minutes_per_review=cfg.MINUTES_PER_REVIEW,
            review_hours_per_analyst=cfg.REVIEW_HOURS_PER_ANALYST_PER_DAY,
            max_analysts=cfg.MAX_ANALYSTS,
            reviewer_accuracy=getattr(cfg, "REVIEWER_ACCURACY", 1.0),
        )

    def cost_table(self) -> np.ndarray:
        """Euros per chart, rows = truth (LONG, SHORT, NONE), cols = decision.

        A reviewed chart costs the analyst's time plus the outcome of their
        call. When the analyst is wrong on a real pattern we count it as a
        missed setup, and on a clean chart as a spurious trade.
        """
        tp, wd, miss = self.cost_correct_trade, self.cost_wrong_direction, self.cost_missed
        fp, tn, rev = self.cost_spurious, self.cost_correct_pass, self.review_cost
        a = self.reviewer_accuracy
        review_pattern = rev + a * tp + (1 - a) * miss
        review_clean = rev + a * tn + (1 - a) * fp
        return np.array([
            [tp, wd, miss, review_pattern],   # truth LONG
            [wd, tp, miss, review_pattern],   # truth SHORT
            [fp, fp, tn, review_clean],       # truth NONE
        ], dtype=float)


class ChartSet:
    """The charts of one split, as arrays."""

    def __init__(self, truth, score, side):
        self.truth = np.asarray(truth, dtype=int)
        self.score = np.asarray(score, dtype=float)
        self.side = np.asarray(side, dtype=int)
        self.pattern = self.truth != NONE
        self._grid = None  # decisions for every (t_low, t_high), computed once

    @classmethod
    def from_records(cls, records: list) -> "ChartSet":
        return cls(
            [r["truth"] for r in records],
            [r["score"] for r in records],
            [r["side"] for r in records],
        )

    def __len__(self) -> int:
        return len(self.truth)

    @property
    def data_base_rate(self) -> float:
        return float(self.pattern.mean()) if len(self) else float("nan")

    def has_both_kinds(self) -> bool:
        return bool(self.pattern.any() and (~self.pattern).any())

    # ---- decisions -------------------------------------------------------

    def decide(self, t_low, t_high) -> np.ndarray:
        """Decision codes. t_low and t_high may be scalars or broadcastable arrays."""
        t_low = np.asarray(t_low, dtype=float)[..., None]
        t_high = np.maximum(np.asarray(t_high, dtype=float)[..., None], t_low)
        return np.where(
            self.score < t_low, NONE,
            np.where(self.score >= t_high, self.side, REVIEW),
        )

    def confusion(self, t_low: float, t_high: float) -> np.ndarray:
        """Counts, rows = truth (LONG, SHORT, NONE), cols = decision (LONG, SHORT, NONE, REVIEW)."""
        d = self.decide(t_low, t_high)
        m = np.zeros((3, 4), dtype=int)
        np.add.at(m, (self.truth, d), 1)
        return m

    # ---- expectations under an assumed base rate -------------------------

    def _expect(self, per_chart: np.ndarray, base_rate: float | None) -> np.ndarray:
        """Average a per-chart quantity, reweighting patterns to the base rate.

        per_chart has the charts on its last axis. Pattern charts get weight
        base_rate in total, pattern-free charts get 1 - base_rate. If the set
        lacks one of the two kinds, reweighting is impossible and the plain
        average is returned.
        """
        if base_rate is None or not self.has_both_kinds():
            return per_chart.mean(axis=-1)
        pos = per_chart[..., self.pattern].mean(axis=-1)
        neg = per_chart[..., ~self.pattern].mean(axis=-1)
        return base_rate * pos + (1.0 - base_rate) * neg

    def effective_base_rate(self, econ: Economics) -> float:
        return self.data_base_rate if econ.base_rate is None else econ.base_rate

    def cost_per_1000(self, econ: Economics, t_low, t_high) -> np.ndarray:
        d = self.decide(t_low, t_high)
        per_chart = econ.cost_table()[self.truth, d]
        return 1000.0 * self._expect(per_chart, self.effective_base_rate(econ))

    def review_share(self, econ: Economics, t_low, t_high) -> np.ndarray:
        d = self.decide(t_low, t_high)
        return self._expect((d == REVIEW).astype(float), self.effective_base_rate(econ))

    def outcomes_per_1000(self, econ: Economics, t_low: float, t_high: float) -> dict:
        """Expected count of each outcome per 1,000 charts."""
        d = self.decide(t_low, t_high)
        t = self.truth
        traded = (d == LONG) | (d == SHORT)
        kinds = {
            "correct trades": traded & (d == t),
            "wrong direction trades": traded & self.pattern & (d != t),
            "spurious trades": traded & ~self.pattern,
            "missed setups": self.pattern & (d == NONE),
            "correct passes": ~self.pattern & (d == NONE),
            "sent to review": d == REVIEW,
        }
        rate = self.effective_base_rate(econ)
        return {k: float(1000.0 * self._expect(v.astype(float), rate)) for k, v in kinds.items()}

    def analysts_needed(self, econ: Economics, share) -> np.ndarray:
        hours = econ.charts_per_day * np.asarray(share) * econ.minutes_per_review / 60.0
        return hours / econ.review_hours_per_analyst

    # ---- operating points -------------------------------------------------

    def sweep_single(self, econ: Economics) -> np.ndarray:
        """Cost per 1,000 charts for each threshold on GRID, no review band."""
        return self.cost_per_1000(econ, GRID, GRID)

    def best_single(self, econ: Economics) -> float:
        """Threshold with the lowest cost. When several tie (a flat bottom),
        take the middle one so small score shifts don't push us off it."""
        costs = self.sweep_single(econ)
        tied = np.flatnonzero(np.isclose(costs, costs.min()))
        return float(GRID[tied[len(tied) // 2]])

    def sweep_band(self, econ: Economics) -> tuple[np.ndarray, np.ndarray]:
        """Cost per 1,000 and review share for every (t_low, t_high) on GRID.

        Returned arrays are indexed [t_low, t_high], and cells with t_low > t_high
        are NaN.
        """
        lo, hi = np.meshgrid(GRID, GRID, indexing="ij")
        if self._grid is None:  # decisions don't depend on costs, so reuse them
            self._grid = self.decide(lo, hi).astype(np.int8)
        rate = self.effective_base_rate(econ)
        cost = 1000.0 * self._expect(econ.cost_table()[self.truth, self._grid], rate)
        share = self._expect((self._grid == REVIEW).astype(float), rate)
        invalid = lo > hi
        cost[invalid] = np.nan
        share[invalid] = np.nan
        return cost, share

    def best_band(self, econ: Economics) -> dict:
        """Cheapest (t_low, t_high) the available analysts can actually staff.

        Ties go first to the option that sends fewer charts to review, then to
        the one closest to the best single threshold.
        """
        cost, share = self.sweep_band(econ)
        t_single = self.best_single(econ)
        unconstrained = np.nanmin(cost)
        feasible = ~np.isnan(cost) & (self.analysts_needed(econ, share) <= econ.max_analysts + 1e-9)
        c = np.where(feasible, cost, np.inf)
        s = np.where(feasible, share, np.inf)
        lo, hi = np.meshgrid(GRID, GRID, indexing="ij")
        dist = np.abs(lo - t_single) + np.abs(hi - t_single)
        order = np.lexsort((dist.ravel(), s.ravel(), np.round(c.ravel(), 6)))
        i, j = np.unravel_index(order[0], c.shape)
        return {
            "t_low": float(GRID[i]),
            "t_high": float(GRID[j]),
            # True when more analysts would buy a cheaper band than this one
            "staffing_bound": bool(c[i, j] > unconstrained + 1e-6),
        }

    # ---- the four ways to run the desk -----------------------------------

    def alternatives(self, econ: Economics, t_single: float, band: tuple) -> list[dict]:
        rows = [
            ("do_nothing", "Do nothing (never trade)", 1.01, 1.01),
            ("analyst_all", "Analyst checks every chart", 0.0, 1.01),
            ("model_alone", f"Model alone (t = {t_single:.2f})", t_single, t_single),
            ("model_band", f"Model + review band ({band[0]:.2f} to {band[1]:.2f})", band[0], band[1]),
        ]
        out = []
        for key, name, lo, hi in rows:
            share = float(self.review_share(econ, lo, hi))
            people = float(self.analysts_needed(econ, share))
            cost = float(self.cost_per_1000(econ, lo, hi))
            out.append({
                "setup": key,
                "option": name,
                "cost_per_1000_eur": round(cost, 2),
                "cost_per_day_eur": round(cost * econ.charts_per_day / 1000.0, 2),
                "review_share": round(share, 4),
                "reviews_per_day": round(share * econ.charts_per_day, 1),
                "analysts_needed": round(people, 2),
                "staffable": people <= econ.max_analysts + 1e-9,
            })
        return out

    # ---- plain classification metrics at one threshold -------------------

    def metrics_at(self, t: float) -> dict:
        d = self.decide(t, t)
        out = {"accuracy": float((d == self.truth).mean()) if len(self) else float("nan")}
        for code, name in ((LONG, "LONG"), (SHORT, "SHORT")):
            predicted = d == code
            actual = self.truth == code
            hit = (predicted & actual).sum()
            out[f"{name}_precision"] = float(hit / predicted.sum()) if predicted.sum() else float("nan")
            out[f"{name}_recall"] = float(hit / actual.sum()) if actual.sum() else float("nan")
        flagged = d != NONE
        hit = (flagged & self.pattern).sum()
        out["pattern_precision"] = float(hit / flagged.sum()) if flagged.sum() else float("nan")
        out["pattern_recall"] = float(hit / self.pattern.sum()) if self.pattern.sum() else float("nan")
        return out


# --------------------------------------------------------------------------
# Sensitivity: how far can each assumption move before the answer changes?
# --------------------------------------------------------------------------

SETUP_NAMES = {
    "do_nothing": "Do nothing",
    "analyst_all": "Analyst checks every chart",
    "model_alone": "Model alone",
    "model_band": "Model + review band",
}
BASE_RATES = np.round(np.arange(0.01, 1.0, 0.01), 2)
MULTIPLIERS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
COST_ASSUMPTIONS = {
    "cost_missed": "Missed setup cost",
    "cost_spurious": "Spurious trade cost",
    "cost_wrong_direction": "Wrong direction cost",
    "review_cost": "Analyst cost per review",
}
ACCURACIES = (0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
ANALYST_COUNTS = (0.5, 1.0, 2.0, 3.0, 5.0)


def compare_setups(val: ChartSet, test: ChartSet, econ: Economics) -> dict:
    """Re-choose thresholds on validation under these assumptions and cost the
    four setups on both splits.

    Which setup to run is decided on validation, like the thresholds: the
    cheapest one the available analysts can staff there. Its cost is then
    reported on test, so the test days are never used to make a choice.
    """
    t_single = val.best_single(econ)
    band = val.best_band(econ)
    pair = (band["t_low"], band["t_high"])
    val_rows = val.alternatives(econ, t_single, pair)
    rows = test.alternatives(econ, t_single, pair)
    staffable = [r for r in val_rows if r["staffable"]]
    best = min(staffable, key=lambda r: r["cost_per_1000_eur"])["setup"]
    best_test = next(r for r in rows if r["setup"] == best)
    return {"t_single": t_single, "band": band, "rows": rows, "val_rows": val_rows,
            "best": best, "best_cost": best_test["cost_per_1000_eur"]}


def _robust_range(rows: list[dict], base_best: str) -> tuple[float, float]:
    """Smallest and largest multiplier, around 1x, at which the best setup is unchanged."""
    ms = [r["multiplier"] for r in rows]
    same = [r["best_setup"] == base_best for r in rows]
    i = ms.index(1.0)
    lo = hi = i
    while lo > 0 and same[lo - 1]:
        lo -= 1
    while hi < len(ms) - 1 and same[hi + 1]:
        hi += 1
    return ms[lo], ms[hi]


def sensitivity(val: ChartSet, test: ChartSet, econ: Economics) -> dict:
    base = compare_setups(val, test, econ)

    # 1. Base rate: cost of each setup as the share of real patterns changes
    by_rate, switches, previous = [], [], None
    for rate in BASE_RATES:
        c = compare_setups(val, test, replace(econ, base_rate=float(rate)))
        row = {"base_rate": float(rate), "best": c["best"]}
        row.update({r["setup"]: r["cost_per_1000_eur"] for r in c["rows"]})
        by_rate.append(row)
        if previous is not None and c["best"] != previous:
            switches.append({"at_base_rate": float(rate), "from": SETUP_NAMES[previous],
                             "to": SETUP_NAMES[c["best"]]})
        previous = c["best"]
    beats_nothing = [r["base_rate"] for r in by_rate if r["model_alone"] < r["do_nothing"] - 1e-9]

    # 2. Each cost assumption scaled up and down, everything else held
    by_cost = []
    for field, label in COST_ASSUMPTIONS.items():
        rows = []
        for m in MULTIPLIERS:
            value = getattr(econ, field) * m
            c = compare_setups(val, test, replace(econ, **{field: value}))
            rows.append({"assumption": label, "multiplier": m, "value": round(value, 2),
                         "best_setup": c["best"], "best_cost_per_1000": c["best_cost"],
                         "t_single": c["t_single"], "band": [c["band"]["t_low"], c["band"]["t_high"]]})
        lo, hi = _robust_range(rows, base["best"])
        by_cost.append({"assumption": label, "base_value": round(getattr(econ, field), 2),
                        "same_answer_from_x": lo, "same_answer_to_x": hi, "rows": rows})

    # 3. Analyst accuracy and analysts available
    by_accuracy = []
    for a in ACCURACIES:
        c = compare_setups(val, test, replace(econ, reviewer_accuracy=a))
        by_accuracy.append({"reviewer_accuracy": a, "best_setup": c["best"],
                            "best_cost_per_1000": c["best_cost"]})
    by_staff = []
    for n in ANALYST_COUNTS:
        c = compare_setups(val, test, replace(econ, max_analysts=n))
        band_row = next(r for r in c["rows"] if r["setup"] == "model_band")
        by_staff.append({"analysts_available": n, "best_setup": c["best"],
                         "best_cost_per_1000": c["best_cost"],
                         "band": [c["band"]["t_low"], c["band"]["t_high"]],
                         "band_review_share": band_row["review_share"]})

    return {
        "base_case_best": base["best"],
        "base_case_cost_per_1000": base["best_cost"],
        "base_rate_reweighting_possible": test.has_both_kinds(),
        "by_base_rate": by_rate,
        "best_setup_switches": switches,
        "model_beats_doing_nothing_from_base_rate": min(beats_nothing) if beats_nothing else None,
        "by_cost": by_cost,
        "by_reviewer_accuracy": by_accuracy,
        "by_analysts_available": by_staff,
    }
