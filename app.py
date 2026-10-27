"""
Assignment 2 dashboard: chart pattern detection for a trading desk.

    uv run python main.py app

Loads the committed weights in weights/ and the outputs train.py saved in
artifacts/. It never trains anything. Costs and staffing are recomputed live
from the saved chart scores with the assumptions in config.py, so moving a
slider redraws everything without touching the model.
"""

import dataclasses
import json
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from PIL import ImageDraw
from ultralytics import YOLO

import config
import costing

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
WEIGHTS = ROOT / "weights" / "best.pt"

for needed in (WEIGHTS, ART / "summary.json", ART / "charts.json"):
    if not needed.exists():
        raise SystemExit(f"Missing {needed.relative_to(ROOT)}. Run `uv run python main.py train` first.")

MODEL = YOLO(str(WEIGHTS))
SUMMARY = json.loads((ART / "summary.json").read_text())
CHARTS = json.loads((ART / "charts.json").read_text())
NAMES = CHARTS["names"]
VAL = costing.ChartSet.from_records(CHARTS["valid"])
TEST = costing.ChartSet.from_records(CHARTS["test"])
BASE_ECON = costing.Economics.from_config(config)
DEFAULT_BASE_RATE = config.BASE_RATE if config.BASE_RATE is not None else round(TEST.data_base_rate, 3)
SIDE_COLOUR = {costing.LONG: (22, 163, 74), costing.SHORT: (220, 38, 38)}


def econ(base_rate: float) -> costing.Economics:
    return dataclasses.replace(BASE_ECON, base_rate=float(base_rate))


@lru_cache(maxsize=64)
def operating_point(base_rate: float) -> tuple[float, float, float]:
    """(single threshold, band low, band high), all chosen on the validation days."""
    e = econ(base_rate)
    band = VAL.best_band(e)
    return VAL.best_single(e), band["t_low"], band["t_high"]


@lru_cache(maxsize=64)
def recommended(base_rate: float) -> str:
    """Cheapest setup the analysts can staff, chosen on the validation days."""
    return costing.SETUP_NAMES[costing.compare_setups(VAL, TEST, econ(base_rate))["best"]]


def clamp(t_low: float, t_high: float) -> tuple[float, float]:
    return float(t_low), float(max(t_low, t_high))


# --------------------------------------------------------------------------
# Decision band: confusion matrix, cost, review load, alternatives
# --------------------------------------------------------------------------

def confusion_frame(t_low: float, t_high: float) -> pd.DataFrame:
    m = TEST.confusion(t_low, t_high)
    frame = pd.DataFrame(m, columns=["Trade LONG", "Trade SHORT", "No trade", "Review"])
    frame.insert(0, "True chart label", ["LONG", "SHORT", "NONE"])
    return frame


def kpis(t_low: float, t_high: float, base_rate: float) -> str:
    e = econ(base_rate)
    cost = float(TEST.cost_per_1000(e, t_low, t_high))
    share = float(TEST.review_share(e, t_low, t_high))
    per_day = share * e.charts_per_day
    people = float(TEST.analysts_needed(e, share))
    ok = "within" if people <= e.max_analysts + 1e-9 else "**above**"
    out = TEST.outcomes_per_1000(e, t_low, t_high)
    t_single, lo, hi = operating_point(base_rate)
    placeholder = ("\n\n> ⚠️ Costs in config.py are still placeholders."
                   if config.COSTS_ARE_PLACEHOLDERS else "")
    return (
        f"### €{cost:,.0f} expected cost per 1,000 charts\n"
        f"**{share:.1%}** of charts go to review: **{per_day:.0f} a day** "
        f"({e.minutes_per_review:g} min each) needs **{people:.2f} analysts**, "
        f"{ok} the {e.max_analysts:g} available.\n\n"
        f"Per 1,000 charts: {out['correct trades']:.0f} correct trades, "
        f"{out['wrong direction trades']:.0f} wrong direction, "
        f"{out['spurious trades']:.0f} spurious, {out['missed setups']:.0f} missed setups.\n\n"
        f"Base rate assumed: {base_rate:.1%} of charts carry a real pattern. "
        f"Chosen on the validation days at this base rate: model alone t = {t_single:.2f}, "
        f"band {lo:.2f} to {hi:.2f}, and the setup to run is **{recommended(base_rate)}**."
        f"{placeholder}"
    )


def alternatives_frame(t_low: float, t_high: float, base_rate: float) -> pd.DataFrame:
    e = econ(base_rate)
    t_single, lo, hi = operating_point(base_rate)
    rows = TEST.alternatives(e, t_single, (lo, hi))
    share = float(TEST.review_share(e, t_low, t_high))
    people = float(TEST.analysts_needed(e, share))
    cost = float(TEST.cost_per_1000(e, t_low, t_high))
    rows.append({
        "option": f"Sliders now ({t_low:.2f} to {t_high:.2f})",
        "cost_per_1000_eur": round(cost, 2),
        "cost_per_day_eur": round(cost * e.charts_per_day / 1000, 2),
        "review_share": round(share, 4),
        "reviews_per_day": round(share * e.charts_per_day, 1),
        "analysts_needed": round(people, 2),
        "staffable": people <= e.max_analysts + 1e-9,
    })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Cost against threshold
# --------------------------------------------------------------------------

def cost_curve(t_low: float, t_high: float, base_rate: float) -> go.Figure:
    e = econ(base_rate)
    t_single, _, _ = operating_point(base_rate)
    test_costs = TEST.sweep_single(e)
    val_costs = VAL.sweep_single(e)
    fig = go.Figure()
    fig.add_vrect(x0=t_low, x1=t_high, fillcolor="#f59e0b", opacity=0.15, line_width=0,
                  annotation_text="review band", annotation_position="top left")
    fig.add_trace(go.Scatter(x=costing.GRID, y=val_costs, name="validation days (used to choose)",
                             line=dict(dash="dot", color="#94a3b8")))
    fig.add_trace(go.Scatter(x=costing.GRID, y=test_costs, name="test days",
                             line=dict(color="#2563eb", width=3)))
    chosen = float(TEST.cost_per_1000(e, t_single, t_single))
    fig.add_trace(go.Scatter(x=[t_single], y=[chosen], mode="markers+text",
                             marker=dict(size=13, color="#dc2626"), name="chosen threshold",
                             text=[f"t = {t_single:.2f}, €{chosen:,.0f}"], textposition="top right"))
    fig.update_layout(
        title="Expected cost per 1,000 charts, model deciding alone",
        xaxis_title="confidence threshold", yaxis_title="€ per 1,000 charts",
        template="plotly_white", height=430, margin=dict(l=60, r=20, t=60, b=50),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


# --------------------------------------------------------------------------
# Per-class results
# --------------------------------------------------------------------------

def per_class_frame() -> pd.DataFrame:
    return pd.DataFrame(SUMMARY["detection_test"]["per_class"])


def baselines_frame() -> pd.DataFrame:
    return pd.DataFrame(SUMMARY["baselines_test"])


def chart_level_frame(t_high: float) -> pd.DataFrame:
    m = TEST.metrics_at(t_high)
    rows = []
    for name in ("LONG", "SHORT"):
        code = costing.TRUTHS.index(name)
        rows.append({
            "chart label": name,
            "test charts": int((TEST.truth == code).sum()),
            "precision": round(m[f"{name}_precision"], 3),
            "recall": round(m[f"{name}_recall"], 3),
        })
    rows.append({
        "chart label": "any pattern (positive class)",
        "test charts": int(TEST.pattern.sum()),
        "precision": round(m["pattern_precision"], 3),
        "recall": round(m["pattern_recall"], 3),
    })
    return pd.DataFrame(rows)


def box_confusion_frame(t: float) -> pd.DataFrame:
    """Box-level confusion on the test days at confidence t (IoU 0.5)."""
    m = costing.box_confusion(CHARTS["test"], len(NAMES), t)
    labels = NAMES + ["background"]
    frame = pd.DataFrame(m, columns=[f"predicted {n}" for n in labels])
    frame.insert(0, "true", labels)
    return frame


def box_per_class_frame(t: float) -> pd.DataFrame:
    m = costing.box_confusion(CHARTS["test"], len(NAMES), t)
    return pd.DataFrame(costing.box_per_class(m, NAMES))


def direction_frame() -> pd.DataFrame:
    rows = []
    for split in ("valid", "test"):
        for r in costing.direction_breakdown(CHARTS[split], NAMES, config.CLASS_SIDE):
            rows.append({"split": split, **r})
    return pd.DataFrame(rows)


def dataset_frame() -> pd.DataFrame:
    rows = []
    for split, d in SUMMARY["dataset"]["description"].items():
        row = {"split": split, "charts": d["charts"], "no pattern": d["charts_with_no_pattern"],
               "both directions": d["charts_with_both_directions"], "tickers": d["distinct_tickers"]}
        row.update({f"{n} boxes": d["instances"][n] for n in NAMES})
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Try a chart
# --------------------------------------------------------------------------

def predict(image, t_low: float, t_high: float):
    if image is None:
        return None, "Upload a chart to see the model's call."
    t_low, t_high = clamp(t_low, t_high)
    result = MODEL.predict(image, conf=config.PREDICT_CONF_FLOOR, imgsz=config.IMGSZ,
                           max_det=config.MAX_BOXES_SAVED_PER_CHART, verbose=False)[0]
    rows = []
    if result.boxes is not None and len(result.boxes):
        for c, s, xy in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                            result.boxes.xyxy.tolist()):
            rows.append((int(c), float(s), xy))
    rows.sort(key=lambda r: -r[1])
    score, side = costing.chart_score([[c, s] for c, s, _ in rows], NAMES, config.CLASS_SIDE)
    decision = costing.decide_one(score, side, t_low, t_high)

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    shown = 0
    for c, s, (x1, y1, x2, y2) in rows:
        if s < t_low:
            continue
        colour = SIDE_COLOUR[costing.side_of(NAMES[c], config.CLASS_SIDE)]
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)
        draw.text((x1 + 4, max(0, y1 - 14)), f"{NAMES[c]} {s:.2f}", fill=colour)
        shown += 1

    verdict = {
        "LONG": "**Trade LONG automatically**",
        "SHORT": "**Trade SHORT automatically**",
        "NONE": "**No trade**, nobody looks at it",
        "REVIEW": "**Send to an analyst** for review",
    }[decision]
    top = f"{NAMES[rows[0][0]]} at {rows[0][1]:.2f}" if rows else "nothing found"
    return canvas, (
        f"{verdict}\n\nMost confident box: {top}. Boxes drawn: {shown} at or above "
        f"t_low = {t_low:.2f}. Green = bullish pattern, red = bearish pattern."
    )


# --------------------------------------------------------------------------
# Sensitivity
# --------------------------------------------------------------------------

SENS = costing.sensitivity(VAL, TEST, BASE_ECON)
SETUP_COLOUR = {"do_nothing": "#64748b", "analyst_all": "#f59e0b",
                "model_alone": "#2563eb", "model_band": "#16a34a"}


def base_rate_figure() -> go.Figure:
    rows = SENS["by_base_rate"]
    x = [r["base_rate"] for r in rows]
    fig = go.Figure()
    for key, name in costing.SETUP_NAMES.items():
        fig.add_trace(go.Scatter(x=x, y=[r[key] for r in rows], name=name,
                                 line=dict(color=SETUP_COLOUR[key], width=2.5)))
    fig.add_vline(x=DEFAULT_BASE_RATE, line_dash="dash", line_color="#dc2626",
                  annotation_text=f"assumed {DEFAULT_BASE_RATE:.0%}", annotation_position="top")
    fig.update_layout(
        title="Cost per 1,000 charts on the test days as the base rate changes",
        xaxis_title="base rate (share of charts with a real pattern)",
        yaxis_title="€ per 1,000 charts", template="plotly_white", height=430,
        margin=dict(l=60, r=20, t=60, b=50), legend=dict(orientation="h", y=-0.2),
    )
    return fig


def sensitivity_text() -> str:
    lines = [f"Base case: run **{costing.SETUP_NAMES[SENS['base_case_best']]}**, "
             f"€{SENS['base_case_cost_per_1000']:,.0f} per 1,000 charts on the test days. "
             "Every row below re-chooses the thresholds and the setup on the validation days."]
    rate = SENS["model_beats_doing_nothing_from_base_rate"]
    lines.append(f"The model alone beats doing nothing from a base rate of {rate:.0%}."
                 if rate is not None else "The model alone never beats doing nothing.")
    for sw in SENS["best_setup_switches"]:
        lines.append(f"At a base rate of {sw['at_base_rate']:.0%} the best setup switches from "
                     f"{sw['from']} to {sw['to']}.")
    if not SENS["base_rate_reweighting_possible"]:
        lines.append("⚠️ The test days lack clean charts, so the base rate cannot be reweighted.")
    return "\n\n".join(lines)


def cost_robustness_frame() -> pd.DataFrame:
    return pd.DataFrame([{
        "assumption": c["assumption"],
        "your value (€)": c["base_value"],
        "same answer from": f"{c['same_answer_from_x']}x",
        "same answer up to": f"{c['same_answer_to_x']}x",
        "best setup at each multiple (0.25x to 4x)": ", ".join(
            costing.SETUP_NAMES[r["best_setup"]] for r in c["rows"]),
    } for c in SENS["by_cost"]])


def accuracy_frame() -> pd.DataFrame:
    return pd.DataFrame([{**r, "best_setup": costing.SETUP_NAMES[r["best_setup"]]}
                         for r in SENS["by_reviewer_accuracy"]])


def staffing_frame() -> pd.DataFrame:
    return pd.DataFrame([{**r, "best_setup": costing.SETUP_NAMES[r["best_setup"]],
                          "band": f"{r['band'][0]:.2f} to {r['band'][1]:.2f}"}
                         for r in SENS["by_analysts_available"]])


def move_frame() -> pd.DataFrame:
    return pd.DataFrame([{
        "expected move": f"{r['expected_move']:.2%}",
        "best setup": costing.SETUP_NAMES[r["best_setup"]],
        "cost per 1,000 charts": r["best_cost_per_1000"],
        "model alone t": r["t_single"],
        "band": f"{r['band'][0]:.2f} to {r['band'][1]:.2f}",
    } for r in SENS.get("by_expected_move", [])])


FAILURES = [(str(ART / "failures" / f["image"]), f["caption"]) for f in SUMMARY.get("failures", [])
            if (ART / "failures" / f["image"]).exists()]


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

def refresh(t_low, t_high, base_rate):
    t_low, t_high = clamp(t_low, t_high)
    return (
        kpis(t_low, t_high, base_rate),
        confusion_frame(t_low, t_high),
        alternatives_frame(t_low, t_high, base_rate),
        cost_curve(t_low, t_high, base_rate),
        chart_level_frame(t_high),
        box_per_class_frame(t_high),
        box_confusion_frame(t_high),
    )


def reset(base_rate):
    _, lo, hi = operating_point(base_rate)
    return lo, hi


_, START_LOW, START_HIGH = operating_point(DEFAULT_BASE_RATE)
ds = SUMMARY["dataset"]
det = SUMMARY["detection_test"]

with gr.Blocks(title="Chart Pattern Desk") as demo:
    gr.Markdown(
        "# Chart pattern desk\n"
        f"YOLO finetuned on [{ds['workspace']}/{ds['project']} v{ds['version']}]({ds['url']}): "
        f"2-minute charts, one stock per day. Test days {', '.join(ds['split']['test_dates'])}, "
        f"never seen in training. Thresholds are chosen on the validation days "
        f"{', '.join(ds['split']['valid_dates'])}."
    )
    with gr.Row():
        t_low = gr.Slider(0, 1, value=START_LOW, step=0.01, label="t_low: below this, no trade")
        t_high = gr.Slider(0, 1, value=START_HIGH, step=0.01, label="t_high: at or above this, trade automatically")
        base_rate = gr.Slider(0.01, 0.99, value=DEFAULT_BASE_RATE, step=0.01,
                              label="Base rate: share of charts with a real pattern")
    reset_btn = gr.Button("Reset to the chosen operating point", size="sm")

    with gr.Tab("Decision band"):
        kpi_md = gr.Markdown()
        gr.Markdown("**Confusion matrix on the test days** (counts of charts, between the "
                    "thresholds a chart goes to review)")
        confusion_df = gr.Dataframe(interactive=False)
        gr.Markdown("**Four ways to run the desk, on the test days**")
        alternatives_df = gr.Dataframe(interactive=False)

    with gr.Tab("Cost against threshold"):
        curve = gr.Plot()

    with gr.Tab("Per-class results"):
        gr.Markdown(f"**Detection on the test days**: mAP50 {det['mAP50']:.3f}, "
                    f"mAP50-95 {det['mAP50-95']:.3f}. Precision and recall per class are at "
                    f"the confidence that maximises F1.")
        gr.Dataframe(per_class_frame(), interactive=False)
        gr.Markdown("**Chart-level calls at t_high** (the positive class is a chart with a "
                    "tradable pattern)")
        chart_df = gr.Dataframe(interactive=False)
        gr.Markdown("**Box-level precision and recall per class at t_high** (a box counts "
                    "as found when it overlaps a true pattern by IoU 0.5 or more)")
        box_class_df = gr.Dataframe(interactive=False)
        gr.Markdown("**Box-level confusion matrix at t_high** (rows are the true class, columns "
                    "the predicted class. The background row counts false alarms on empty chart "
                    "area, the background column counts true patterns the model missed)")
        box_cm_df = gr.Dataframe(interactive=False)
        gr.Markdown("**Baselines on the same test days**")
        gr.Dataframe(baselines_frame(), interactive=False)

    with gr.Tab("Where it fails"):
        gr.Markdown("The model's worst mistakes on the test days at the chosen threshold. "
                    "Blue boxes are the true patterns, green and red are the model's top boxes "
                    "(bullish and bearish).")
        if FAILURES:
            gr.Gallery(value=FAILURES, columns=3, height="auto", show_label=False)
        else:
            gr.Markdown("No failures were saved by the training run.")
        gr.Markdown("**Does the model's most confident box point the right way?** Charts whose "
                    "patterns all point one way, against charts with both a bullish and a bearish "
                    "pattern. On a single-direction chart the most recent pattern is simply its "
                    "direction.")
        gr.Dataframe(direction_frame(), interactive=False)

    with gr.Tab("Sensitivity"):
        gr.Markdown(sensitivity_text())
        gr.Plot(base_rate_figure())
        gr.Markdown("**How far each cost can move before the recommended setup changes** "
                    "(at the assumed base rate)")
        gr.Dataframe(cost_robustness_frame(), interactive=False)
        if SENS.get("by_expected_move"):
            gr.Markdown(f"**Expected move on a real pattern** (yours: {config.EXPECTED_MOVE:.2%}, "
                        f"trading costs {config.ROUND_TRIP_COST:.2%}). It sets both the "
                        "missed setup and the wrong direction cost.")
            gr.Dataframe(move_frame(), interactive=False)
        gr.Markdown("**Analyst accuracy on reviewed charts**")
        gr.Dataframe(accuracy_frame(), interactive=False)
        gr.Markdown("**Analysts available for the review queue**")
        gr.Dataframe(staffing_frame(), interactive=False)

    with gr.Tab("Try a chart"):
        with gr.Row():
            upload = gr.Image(type="pil", label="Upload a chart")
            annotated = gr.Image(type="pil", label="Model output", interactive=False)
        verdict = gr.Markdown()
        run_btn = gr.Button("Run the model", variant="primary")
        examples = [str(ART / "examples" / f) for f in SUMMARY.get("examples", [])
                    if (ART / "examples" / f).exists()]
        if examples:
            gr.Examples(examples=examples, inputs=upload)

    with gr.Tab("Dataset"):
        gr.Markdown(f"Split by trading date. Train {', '.join(ds['split']['train_dates'])}.")
        gr.Dataframe(dataset_frame(), interactive=False)

    outputs = [kpi_md, confusion_df, alternatives_df, curve, chart_df, box_class_df, box_cm_df]
    for control in (t_low, t_high, base_rate):
        control.change(refresh, [t_low, t_high, base_rate], outputs)
    reset_btn.click(reset, [base_rate], [t_low, t_high])
    run_btn.click(predict, [upload, t_low, t_high], [annotated, verdict])
    upload.change(predict, [upload, t_low, t_high], [annotated, verdict])
    demo.load(refresh, [t_low, t_high, base_rate], outputs)


if __name__ == "__main__":
    demo.launch()
