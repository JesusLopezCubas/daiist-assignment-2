"""
Assignment 2 training pipeline: chart pattern detection for a trading desk.

    uv run python main.py train

Start to finish, with no manual steps:

1. Download the dataset version named in config.py from Roboflow Universe.
2. Split it by trading date, because the published version has no
   validation or test split.
3. Describe the dataset: split sizes, instances per class, charts with no
   pattern, box size relative to the chart.
4. Finetune the COCO checkpoint on the training days.
5. Evaluate on the test days against two baselines: the same COCO checkpoint
   used zero-shot, and a model that predicts nothing.
6. Score every validation and test chart, choose the operating point on the
   validation days, and report its cost on the test days.
7. Save weights/best.pt and everything app.py needs in artifacts/.

All assumptions live in config.py.
"""

import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

import config
import costing
from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
RAW_DIR = DATASETS / f"{config.ROBOFLOW_PROJECT}-v{config.ROBOFLOW_VERSION}"
SPLIT_DIR = DATASETS / "split-by-date"
RUNS = ROOT / "runs"
WEIGHTS = ROOT / "weights"
ARTIFACTS = ROOT / "artifacts"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")

# "uber_2m_2023-09-18_to_2023-09-19" once Roboflow's ".rf.<hash>" and "_png" are removed
CHART_NAME = re.compile(
    r"^(?P<ticker>.+?)_(?P<timeframe>\d+[mhd])_(?P<date>\d{4}-\d{2}-\d{2})_to_\d{4}-\d{2}-\d{2}"
)


# --------------------------------------------------------------------------
# 1. Download
# --------------------------------------------------------------------------

def download_dataset() -> Path:
    if (RAW_DIR / "data.yaml").exists():
        print(f"Dataset already downloaded at {RAW_DIR}")
        return RAW_DIR
    from roboflow import Roboflow

    rf = Roboflow(api_key=load_api_key())
    version = (
        rf.workspace(config.ROBOFLOW_WORKSPACE)
        .project(config.ROBOFLOW_PROJECT)
        .version(config.ROBOFLOW_VERSION)
    )
    version.download("yolov11", location=str(RAW_DIR))
    return RAW_DIR


def class_names(dataset_dir: Path) -> list[str]:
    names = yaml.safe_load((dataset_dir / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return list(names)


# --------------------------------------------------------------------------
# 2. Split by trading date
# --------------------------------------------------------------------------

def original_stem(image: Path) -> str:
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", image.stem)
    return re.sub(r"_(png|jpe?g|webp|bmp)$", "", stem, flags=re.IGNORECASE)


def parse_chart(image: Path) -> tuple[str | None, str | None]:
    """(ticker, trading date) from the file name, or (None, None)."""
    match = CHART_NAME.match(original_stem(image))
    if not match:
        return None, None
    return match["ticker"].lower(), match["date"]


def all_images(dataset_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for split in SPLITS:
        img_dir = dataset_dir / split / "images"
        if img_dir.is_dir():
            for image in sorted(img_dir.iterdir()):
                if image.suffix.lower() in IMAGE_EXTS:
                    pairs.append((image, dataset_dir / split / "labels" / f"{image.stem}.txt"))
    return pairs


def split_by_date(raw_dir: Path, names: list[str]) -> tuple[Path, dict]:
    pairs = all_images(raw_dir)
    dated = [(img, lbl, *parse_chart(img)) for img, lbl in pairs]
    dates = sorted({d for *_, d in dated if d})
    needed = config.TEST_DATES + config.VALID_DATES + 1
    if len(dates) < needed:
        raise RuntimeError(
            f"Only {len(dates)} trading dates parsed from file names, but the split needs at least "
            f"{needed} for the split in config.py."
        )
    test_dates = dates[-config.TEST_DATES:]
    valid_dates = dates[-(config.TEST_DATES + config.VALID_DATES):-config.TEST_DATES]

    def split_of(date):
        if date in test_dates:
            return "test"
        if date in valid_dates:
            return "valid"
        return "train"  # includes the few names we cannot date, if any

    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)
    for split in SPLITS:
        (SPLIT_DIR / split / "images").mkdir(parents=True)
        (SPLIT_DIR / split / "labels").mkdir(parents=True)

    undated = 0
    for image, label, _, date in dated:
        undated += date is None
        split = split_of(date)
        shutil.copy2(image, SPLIT_DIR / split / "images" / image.name)
        target = SPLIT_DIR / split / "labels" / f"{image.stem}.txt"
        if label.exists():
            shutil.copy2(label, target)
        else:
            target.write_text("")

    (SPLIT_DIR / "data.yaml").write_text(yaml.safe_dump({
        "path": str(SPLIT_DIR),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": names,
    }, sort_keys=False))

    info = {
        "all_dates": dates,
        "train_dates": [d for d in dates if split_of(d) == "train"],
        "valid_dates": valid_dates,
        "test_dates": test_dates,
        "undated_images_put_in_train": undated,
    }
    print(f"Split by date: train {info['train_dates']}, valid {valid_dates}, test {test_dates}")
    return SPLIT_DIR, info


# --------------------------------------------------------------------------
# 3. Describe the dataset
# --------------------------------------------------------------------------

def read_boxes(label: Path) -> list[list[float]]:
    """Ground truth as [cls, x1, y1, x2, y2] in 0..1 image coordinates."""
    boxes = []
    if not label.exists():
        return boxes
    for row in label.read_text().splitlines():
        parts = row.split()
        if not parts:
            continue
        cls, values = int(parts[0]), [float(v) for v in parts[1:]]
        if len(values) == 4:
            cx, cy, w, h = values
            boxes.append([cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        else:  # polygon
            xs, ys = values[0::2], values[1::2]
            boxes.append([cls, min(xs), min(ys), max(xs), max(ys)])
    return boxes


def split_pairs(split_dir: Path, split: str) -> list[tuple[Path, Path]]:
    img_dir = split_dir / split / "images"
    return [
        (image, split_dir / split / "labels" / f"{image.stem}.txt")
        for image in sorted(img_dir.iterdir())
        if image.suffix.lower() in IMAGE_EXTS
    ]


def describe(split_dir: Path, names: list[str]) -> dict:
    out = {}
    for split in SPLITS:
        pairs = split_pairs(split_dir, split)
        instances, charts_with = Counter(), Counter()
        truths, areas, widths = Counter(), [], []
        tickers, mixed, empty = set(), 0, 0
        for image, label in pairs:
            boxes = read_boxes(label)
            ticker, _ = parse_chart(image)
            tickers.add(ticker)
            empty += not boxes
            truth = costing.chart_truth(boxes, names, config.CLASS_SIDE, config.MIXED_CHART_RULE)
            truths[costing.TRUTHS[truth]] += 1
            sides = {costing.side_of(names[int(b[0])], config.CLASS_SIDE) for b in boxes}
            mixed += len(sides) > 1
            for name in {names[int(b[0])] for b in boxes}:
                charts_with[name] += 1
            for b in boxes:
                instances[names[int(b[0])]] += 1
                w, h = b[3] - b[1], b[4] - b[2]
                areas.append(w * h)
                widths.append(w)

        def q(values):
            return [round(float(np.quantile(values, p)), 3) for p in (0.1, 0.5, 0.9)] if values else None

        out[split] = {
            "charts": len(pairs),
            "charts_with_no_pattern": empty,
            "charts_with_both_directions": mixed,
            "chart_truth": {k: truths.get(k, 0) for k in costing.TRUTHS},
            "instances": {n: instances.get(n, 0) for n in names},
            "charts_containing": {n: charts_with.get(n, 0) for n in names},
            "distinct_tickers": len(tickers - {None}),
            "box_area_share_p10_p50_p90": q(areas),
            "box_width_share_p10_p50_p90": q(widths),
        }
    return out


def print_description(desc: dict, names: list[str]) -> None:
    print("\nDataset description")
    for split, d in desc.items():
        print(f"  [{split}] {d['charts']} charts, {d['charts_with_no_pattern']} with no pattern, "
              f"{d['charts_with_both_directions']} with both directions, "
              f"{d['distinct_tickers']} tickers")
        print(f"     chart truth: {d['chart_truth']}")
        for n in names:
            print(f"     {n:<22} {d['instances'][n]:>5} boxes in {d['charts_containing'][n]:>4} charts")
        print(f"     box area / chart p10,p50,p90: {d['box_area_share_p10_p50_p90']}  "
              f"box width / chart: {d['box_width_share_p10_p50_p90']}")


# --------------------------------------------------------------------------
# 4. Finetune
# --------------------------------------------------------------------------

def finetune(data_yaml: Path) -> tuple[Path, dict]:
    from ultralytics import YOLO

    model = YOLO(config.BASE_MODEL)
    started = time.time()
    model.train(
        data=str(data_yaml),
        epochs=config.EPOCHS,
        patience=config.PATIENCE,
        imgsz=config.IMGSZ,
        batch=config.BATCH,
        seed=config.SEED,
        fliplr=config.FLIPLR,
        flipud=config.FLIPUD,
        project=str(RUNS),
        name="finetune",
        exist_ok=True,
        verbose=False,
    )
    trainer = model.trainer
    best = Path(trainer.best)
    WEIGHTS.mkdir(exist_ok=True)
    shutil.copy2(best, WEIGHTS / "best.pt")

    import torch
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    info = {
        "base_model": config.BASE_MODEL,
        "epochs_requested": config.EPOCHS,
        "epochs_run": int(trainer.epoch) + 1,
        "imgsz": config.IMGSZ,
        "batch": config.BATCH,
        "patience": config.PATIENCE,
        "seed": config.SEED,
        "fliplr": config.FLIPLR,
        "flipud": config.FLIPUD,
        "device": device,
        "minutes": round((time.time() - started) / 60, 1),
    }
    print(f"\nTrained {info['epochs_run']} epochs on {device} in {info['minutes']} min. "
          f"Best weights copied to weights/best.pt")
    return WEIGHTS / "best.pt", info


# --------------------------------------------------------------------------
# 5. Detection metrics on the test days
# --------------------------------------------------------------------------

def detection_metrics(weights: Path, data_yaml: Path, names: list[str], desc: dict) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml), split="test", imgsz=config.IMGSZ, batch=config.BATCH,
        conf=config.PREDICT_CONF_FLOOR, plots=True, project=str(RUNS), name="test_eval",
        exist_ok=True, verbose=False,
    )
    box = metrics.box
    per_class = []
    for k, cls in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(k)
        name = names[int(cls)]
        per_class.append({
            "class": name,
            "direction": config.CLASS_SIDE.get(costing.normalise(name), "?"),
            "test_instances": desc["test"]["instances"].get(name, 0),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "AP50": round(float(ap50), 4),
            "AP50-95": round(float(ap), 4),
        })
    seen = {row["class"] for row in per_class}
    for name in names:
        if name not in seen:
            per_class.append({"class": name, "test_instances": 0, "note": "no test instances"})

    plots = ARTIFACTS / "test_eval"
    plots.mkdir(parents=True, exist_ok=True)
    for png in Path(metrics.save_dir).glob("*.png"):
        shutil.copy2(png, plots / png.name)

    return {
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "mean_precision": round(float(box.mp), 4),
        "mean_recall": round(float(box.mr), 4),
        "per_class": per_class,
        "box_confusion_matrix": {
            "labels": names + ["background"],
            "matrix": metrics.confusion_matrix.matrix.astype(int).tolist(),
            "orientation": "rows = predicted class, columns = true class (Ultralytics convention)",
        },
        "note": "precision and recall per class are at the confidence that maximises F1",
    }


# --------------------------------------------------------------------------
# 6. Chart-level scores for the cost analysis
# --------------------------------------------------------------------------

def predict_boxes(model, image: Path) -> list[list[float]]:
    """[cls, conf, x1, y1, x2, y2] for the most confident boxes on one chart."""
    result = model.predict(
        str(image), conf=config.PREDICT_CONF_FLOOR, imgsz=config.IMGSZ,
        max_det=config.MAX_BOXES_SAVED_PER_CHART, verbose=False,
    )[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxyn.cpu().numpy()
    rows = [[int(c), round(float(s), 4), *[round(float(v), 4) for v in xy]]
            for c, s, xy in zip(cls, conf, xyxy)]
    return sorted(rows, key=lambda r: -r[1])


def score_charts(split_dir: Path, split: str, names: list[str], tuned, coco, coco_side: int) -> list[dict]:
    records = []
    for image, label in split_pairs(split_dir, split):
        gt = read_boxes(label)
        pred = predict_boxes(tuned, image)
        coco_pred = predict_boxes(coco, image)
        score, side = costing.chart_score(pred, names, config.CLASS_SIDE)
        ticker, date = parse_chart(image)
        records.append({
            "file": image.name,
            "ticker": ticker,
            "date": date,
            "truth": costing.chart_truth(gt, names, config.CLASS_SIDE, config.MIXED_CHART_RULE),
            "score": score,
            "side": side,
            # COCO knows no chart patterns, so it can only say "something is
            # here". It gets the more common direction in training as its side.
            "coco_score": coco_pred[0][1] if coco_pred else 0.0,
            "coco_side": coco_side,
            "coco_classes": [coco.names[int(b[0])] for b in coco_pred[:3]],
            "gt": [[int(b[0])] + [round(v, 4) for v in b[1:]] for b in gt],
            "pred": pred,
            "coco_pred": coco_pred,
        })
    return records


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def class_agnostic_ap50(records: list[dict], key: str) -> float:
    """AP at IoU 0.5 ignoring class: does the model put a box on the pattern at all?
    The only box metric on which COCO and the finetuned model can be compared."""
    total_gt = sum(len(r["gt"]) for r in records)
    if total_gt == 0:
        return float("nan")
    dets = [(p[1], i, p[2:]) for i, r in enumerate(records) for p in r[key]]
    dets.sort(key=lambda d: -d[0])
    used = [np.zeros(len(r["gt"]), dtype=bool) for r in records]
    tp = np.zeros(len(dets))
    for k, (_, i, box) in enumerate(dets):
        gts = records[i]["gt"]
        if not gts:
            continue
        overlaps = [iou(box, g[1:]) if not used[i][j] else 0.0 for j, g in enumerate(gts)]
        j = int(np.argmax(overlaps))
        if overlaps[j] >= 0.5:
            used[i][j] = True
            tp[k] = 1
    if not len(dets):
        return 0.0
    cum_tp = np.cumsum(tp)
    recall = cum_tp / total_gt
    precision = cum_tp / np.arange(1, len(dets) + 1)
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[1.0], precision, [0.0]])
    p = np.maximum.accumulate(p[::-1])[::-1]
    return float(np.sum((r[1:] - r[:-1]) * p[1:]))


# --------------------------------------------------------------------------
# 7. Put it together
# --------------------------------------------------------------------------

def chartset(records: list[dict], prefix: str = "") -> costing.ChartSet:
    return costing.ChartSet(
        [r["truth"] for r in records],
        [r[f"{prefix}score"] for r in records],
        [r[f"{prefix}side"] for r in records],
    )


def baselines_table(val, test, econ, desc, det) -> list[dict]:
    rows = []
    # Finetuned and COCO each get their own best threshold, chosen on validation.
    for label, prefix, ap_key, map50 in (
        ("Finetuned " + config.BASE_MODEL.replace(".pt", ""), "", "pred", det["mAP50"]),
        ("COCO " + config.BASE_MODEL.replace(".pt", "") + " zero-shot", "coco_", "coco_pred", 0.0),
    ):
        v, t = chartset(val, prefix), chartset(test, prefix)
        th = v.best_single(econ)
        m = t.metrics_at(th)
        rows.append({
            "model": label,
            "class_mAP50": map50,
            "class_agnostic_AP50": round(class_agnostic_ap50(test, ap_key), 4),
            "threshold_from_valid": th,
            "direction_accuracy": round(m["accuracy"], 4),
            "pattern_precision": round(m["pattern_precision"], 4),
            "pattern_recall": round(m["pattern_recall"], 4),
            "cost_per_1000_eur": round(float(t.cost_per_1000(econ, th, th)), 2),
        })
    nothing = costing.ChartSet([r["truth"] for r in test], [0.0] * len(test), [0] * len(test))
    m = nothing.metrics_at(0.5)
    rows.append({
        "model": "Predict nothing",
        "class_mAP50": 0.0,
        "class_agnostic_AP50": 0.0,
        "threshold_from_valid": None,
        "direction_accuracy": round(m["accuracy"], 4),
        "pattern_precision": float("nan"),
        "pattern_recall": 0.0,
        "cost_per_1000_eur": round(float(nothing.cost_per_1000(econ, 0.5, 0.5)), 2),
    })
    majority = Counter(desc["train"]["chart_truth"]).most_common(1)[0][0]
    if majority != "NONE":
        code = costing.TRUTHS.index(majority)
        always = costing.ChartSet([r["truth"] for r in test], [1.0] * len(test), [code] * len(test))
        m = always.metrics_at(0.5)
        rows.append({
            "model": f"Always {majority} (majority chart label)",
            "class_mAP50": 0.0,
            "class_agnostic_AP50": 0.0,
            "threshold_from_valid": None,
            "direction_accuracy": round(m["accuracy"], 4),
            "pattern_precision": round(m["pattern_precision"], 4),
            "pattern_recall": round(m["pattern_recall"], 4),
            "cost_per_1000_eur": round(float(always.cost_per_1000(econ, 0.5, 0.5)), 2),
        })
    return rows


def save_examples(records: list[dict]) -> list[str]:
    """Copy a few test charts for the dashboard (dataset is CC BY 4.0)."""
    out_dir = ARTIFACTS / "examples"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    picked, seen = [], set()
    for r in records:
        if r["truth"] not in seen:
            seen.add(r["truth"])
            picked.append(r)
    for r in records:
        if len(picked) >= 4:
            break
        if r not in picked:
            picked.append(r)
    names = []
    for r in picked[:4]:
        shutil.copy2(SPLIT_DIR / "test" / "images" / r["file"], out_dir / r["file"])
        names.append(r["file"])
    return names


def render_chart(image_path: Path, record: dict, names: list[str], top: int = 3):
    """The chart with its true patterns in blue and the model's top boxes in
    green (bullish) or red (bearish)."""
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for cls, x1, y1, x2, y2 in record["gt"]:
        draw.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=(37, 99, 235), width=2)
        draw.text((x1 * w + 3, y2 * h - 12), f"true: {names[cls]}", fill=(37, 99, 235))
    for cls, conf, x1, y1, x2, y2 in record["pred"][:top]:
        side = costing.side_of(names[cls], config.CLASS_SIDE)
        colour = (22, 163, 74) if side == costing.LONG else (220, 38, 38)
        draw.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=colour, width=2)
        draw.text((x1 * w + 3, max(0, y1 * h - 12)), f"{names[cls]} {conf:.2f}", fill=colour)
    return img


def save_failures(records: list[dict], names: list[str], t: float, per_kind: int = 3) -> list[dict]:
    """The model's worst test-day mistakes at the chosen threshold, as images."""
    out_dir = ARTIFACTS / "failures"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    label = costing.TRUTHS

    def decision(r):
        return costing.decide_one(r["score"], r["side"], t, t)

    # Each list is ranked by how badly the model got it, whether or not the
    # chosen threshold happened to act on it, so the gallery is never empty.
    kinds = {
        # real pattern, but the model's top box points the other way
        "wrong direction": sorted(
            [r for r in records if r["truth"] != costing.NONE and r["pred"] and r["side"] != r["truth"]],
            key=lambda r: -r["score"]),
        # charts with no pattern where the model is most sure it sees one
        "confident on a clean chart": sorted(
            [r for r in records if r["truth"] == costing.NONE], key=lambda r: -r["score"]),
        # real patterns the model scored lowest
        "missed pattern": sorted(
            [r for r in records if r["truth"] != costing.NONE], key=lambda r: r["score"]),
    }
    saved = []
    for kind, rows in kinds.items():
        for i, r in enumerate(rows[:per_kind]):
            name = f"{kind.replace(' ', '_')}_{i + 1}.png"
            render_chart(SPLIT_DIR / "test" / "images" / r["file"], r, names).save(out_dir / name)
            top = f"{names[r['pred'][0][0]]} at {r['pred'][0][1]:.2f}" if r["pred"] else "no box"
            saved.append({
                "kind": kind,
                "image": name,
                "caption": (f"{kind}: {r['ticker']} {r['date']}, truth {label[r['truth']]}, "
                            f"model says {decision(r)} (top box {top})"),
            })
    return saved


def main() -> None:
    if config.COSTS_ARE_PLACEHOLDERS:
        print("\n" + "!" * 78)
        print("  config.py still holds PLACEHOLDER costs. Replace them with your own figures.")
        print("!" * 78 + "\n")

    raw = download_dataset()
    names = class_names(raw)
    for n in names:
        costing.side_of(n, config.CLASS_SIDE)  # fail early if a class has no direction

    split_dir, split_info = split_by_date(raw, names)
    desc = describe(split_dir, names)
    print_description(desc, names)

    weights, train_info = finetune(split_dir / "data.yaml")
    det = detection_metrics(weights, split_dir / "data.yaml", names, desc)

    from ultralytics import YOLO
    tuned, coco = YOLO(str(weights)), YOLO(config.BASE_MODEL)
    majority_side = costing.LONG if desc["train"]["chart_truth"]["LONG"] >= desc["train"]["chart_truth"]["SHORT"] else costing.SHORT
    val = score_charts(split_dir, "valid", names, tuned, coco, majority_side)
    test = score_charts(split_dir, "test", names, tuned, coco, majority_side)

    econ = costing.Economics.from_config(config)
    v, t = chartset(val), chartset(test)
    t_single = v.best_single(econ)
    band = v.best_band(econ)
    lo, hi = band["t_low"], band["t_high"]

    choice = costing.compare_setups(v, t, econ)
    sens = costing.sensitivity(v, t, econ)

    ARTIFACTS.mkdir(exist_ok=True)
    examples = save_examples(test)
    failures = save_failures(test, names, t_single)
    (ARTIFACTS / "charts.json").write_text(json.dumps({"names": names, "valid": val, "test": test}))
    (ARTIFACTS / "sensitivity.json").write_text(json.dumps(sens, indent=1, default=float))

    summary = {
        "dataset": {
            "workspace": config.ROBOFLOW_WORKSPACE,
            "project": config.ROBOFLOW_PROJECT,
            "version": config.ROBOFLOW_VERSION,
            "url": f"https://universe.roboflow.com/{config.ROBOFLOW_WORKSPACE}/{config.ROBOFLOW_PROJECT}/dataset/{config.ROBOFLOW_VERSION}",
            "classes": names,
            "split": split_info,
            "description": desc,
        },
        "training": train_info,
        "detection_test": det,
        "baselines_test": baselines_table(val, test, econ, desc, det),
        "operating_point": {
            "chosen_on": "valid",
            "single_threshold": t_single,
            "band": band,
            "base_rate_used": t.effective_base_rate(econ),
            "base_rate_in_test_data": t.data_base_rate,
            "base_rate_reweighting_possible": t.has_both_kinds(),
            "recommended_setup": costing.SETUP_NAMES[choice["best"]],
            "recommended_setup_note": "cheapest setup the analysts can staff on the validation days",
        },
        "test_results": {
            "model_alone": {
                "cost_per_1000_eur": round(float(t.cost_per_1000(econ, t_single, t_single)), 2),
                "outcomes_per_1000": t.outcomes_per_1000(econ, t_single, t_single),
                "confusion": t.confusion(t_single, t_single).tolist(),
                "metrics": t.metrics_at(t_single),
            },
            "model_plus_review": {
                "cost_per_1000_eur": round(float(t.cost_per_1000(econ, lo, hi)), 2),
                "review_share": round(float(t.review_share(econ, lo, hi)), 4),
                "outcomes_per_1000": t.outcomes_per_1000(econ, lo, hi),
                "confusion": t.confusion(lo, hi).tolist(),
            },
            "alternatives": t.alternatives(econ, t_single, (lo, hi)),
        },
        "economics_used": {k: getattr(config, k) for k in dir(config) if k.isupper() and (
            k.startswith("COST") or k in {"BASE_RATE", "CHARTS_PER_DAY", "MINUTES_PER_REVIEW",
                                          "ANALYST_COST_PER_HOUR", "REVIEW_HOURS_PER_ANALYST_PER_DAY",
                                          "MAX_ANALYSTS", "REVIEWER_ACCURACY"})},
        "examples": examples,
        "failures": failures,
    }
    (ARTIFACTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print("\nBaselines on the test days")
    for row in summary["baselines_test"]:
        print("  ", row)
    print(f"\nOperating point chosen on validation: model alone t = {t_single:.2f}, "
          f"band {lo:.2f} to {hi:.2f}")
    print("Alternatives on the test days")
    for row in summary["test_results"]["alternatives"]:
        print("  ", row)
    print(f"\nRecommended setup (chosen on validation): {costing.SETUP_NAMES[choice['best']]}, "
          f"€{choice['best_cost']:,.0f} per 1,000 charts on the test days")
    print("\nSensitivity (thresholds and setup re-chosen on validation each time)")
    rate = sens["model_beats_doing_nothing_from_base_rate"]
    print(f"   model alone beats doing nothing from a base rate of {rate}" if rate is not None
          else "   model alone never beats doing nothing at any base rate")
    for sw in sens["best_setup_switches"]:
        print(f"   at base rate {sw['at_base_rate']:.2f} the best setup switches from {sw['from']} to {sw['to']}")
    for c in sens["by_cost"]:
        print(f"   {c['assumption']} (€{c['base_value']:,.2f}): same answer from "
              f"{c['same_answer_from_x']}x to {c['same_answer_to_x']}x")
    if not t.has_both_kinds():
        print("\nWARNING: the test days lack either pattern or pattern-free charts, "
              "so costs could not be reweighted to BASE_RATE.")
    print(f"\nSaved weights/best.pt and artifacts/. Run `uv run python main.py app` next.")


if __name__ == "__main__":
    main()
