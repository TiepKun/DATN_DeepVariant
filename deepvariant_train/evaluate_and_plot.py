from __future__ import annotations

import argparse
import csv
import html
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf

from deepvariant_train.data import (
    DEFAULT_TEST_PATTERN,
    DEFAULT_TRAIN_PATTERN,
    DEFAULT_VAL_PATTERN,
    infer_input_shape,
    list_tfrecords,
    parse_input_shape,
)
from deepvariant_train.models import MODEL_BUILDERS

DEFAULT_MODEL_ORDER = ("inceptionv3", "convnextv2_tiny", "efficientnetv2_s", "vit_tiny")
DEFAULT_MODEL_LABELS = {
    "inceptionv3": "InceptionV3",
    "convnextv2_tiny": "ConvNeXtV2 Tiny",
    "efficientnetv2_s": "EfficientNetV2 S",
    "vit_tiny": "ViT Tiny",
}
DEFAULT_VARIANT_TYPE_NAMES = {
    -1: "UNKNOWN",
    1: "SNP",
    2: "INDEL",
}
PALETTE = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained models by SNP/INDEL and create charts."
    )
    parser.add_argument("--run-dir", default="runs/gpu-terminal-progress")
    parser.add_argument("--data-dir", default="tfrecords")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--pattern", default=None, help="Override TFRecord glob pattern.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    parser.add_argument("--models", default=",".join(DEFAULT_MODEL_ORDER))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-shape", default=None)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--class-names", default="0,1,2")
    parser.add_argument("--image-key", default="image/encoded")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--variant-type-key", default="variant_type")
    parser.add_argument("--shape-key", default="image/shape")
    parser.add_argument("--compression-type", default="GZIP")
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


def split_pattern(split: str) -> str:
    if split == "train":
        return DEFAULT_TRAIN_PATTERN
    if split == "val":
        return DEFAULT_VAL_PATTERN
    if split == "test":
        return DEFAULT_TEST_PATTERN
    raise ValueError(f"Unsupported split: {split}")


def read_history(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: float(value) for key, value in row.items() if value != ""})
    return rows


def history_summary(model_name: str, rows: list[dict[str, float]]) -> dict[str, float | str | int]:
    if not rows:
        return {"model": model_name}
    best_acc_row = max(rows, key=lambda row: row.get("val_accuracy", float("-inf")))
    best_loss_row = min(rows, key=lambda row: row.get("val_loss", float("inf")))
    last = rows[-1]
    return {
        "model": model_name,
        "epochs": len(rows),
        "best_val_accuracy": best_acc_row.get("val_accuracy", math.nan),
        "best_val_accuracy_epoch": int(best_acc_row.get("epoch", len(rows) - 1)) + 1,
        "best_val_loss": best_loss_row.get("val_loss", math.nan),
        "best_val_loss_epoch": int(best_loss_row.get("epoch", len(rows) - 1)) + 1,
        "last_accuracy": last.get("accuracy", math.nan),
        "last_loss": last.get("loss", math.nan),
        "last_val_accuracy": last.get("val_accuracy", math.nan),
        "last_val_loss": last.get("val_loss", math.nan),
    }


def make_eval_dataset(
    files: Iterable[str | Path],
    *,
    input_shape: tuple[int, int, int],
    batch_size: int,
    compression_type: str,
    image_key: str,
    label_key: str,
    variant_type_key: str,
    shape_key: str,
    normalize: bool,
) -> tf.data.Dataset:
    file_list = [str(Path(path)) for path in files]
    expected_size = int(np.prod(input_shape))
    feature_spec = {
        image_key: tf.io.FixedLenFeature([], tf.string),
        label_key: tf.io.FixedLenFeature([], tf.int64),
        variant_type_key: tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    }
    if shape_key:
        feature_spec[shape_key] = tf.io.FixedLenFeature(
            [3], tf.int64, default_value=list(input_shape)
        )

    def parse(serialized: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        features = tf.io.parse_single_example(serialized, feature_spec)
        image = tf.io.decode_raw(features[image_key], out_type=tf.uint8)
        image = tf.ensure_shape(image, [expected_size])
        image = tf.reshape(image, input_shape)
        image = tf.cast(image, tf.float32)
        if normalize:
            image = image / 255.0
        label = tf.cast(features[label_key], tf.int32)
        variant_type = tf.cast(features[variant_type_key], tf.int32)
        return image, label, variant_type

    dataset = tf.data.TFRecordDataset(file_list, compression_type=compression_type)
    dataset = dataset.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def empty_eval_state(num_classes: int) -> dict[int | str, np.ndarray]:
    return {
        "ALL": np.zeros((num_classes, num_classes), dtype=np.int64),
        1: np.zeros((num_classes, num_classes), dtype=np.int64),
        2: np.zeros((num_classes, num_classes), dtype=np.int64),
    }


def update_confusion(
    matrix: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    num_classes: int,
) -> None:
    for label, prediction in zip(labels.tolist(), predictions.tolist()):
        if 0 <= label < num_classes and 0 <= prediction < num_classes:
            matrix[label, prediction] += 1


def evaluate_model(
    model_path: Path,
    dataset: tf.data.Dataset,
    *,
    num_classes: int,
) -> tuple[dict[int | str, np.ndarray], dict[str, float | int]]:
    load_start = time.perf_counter()
    model = tf.keras.models.load_model(model_path, compile=False)
    load_time_sec = time.perf_counter() - load_start
    state = empty_eval_state(num_classes)
    predict_time_sec = 0.0
    total_samples = 0

    for images, labels, variant_types in dataset:
        batch_size = int(images.shape[0]) if images.shape[0] is not None else int(tf.shape(images)[0])
        predict_start = time.perf_counter()
        logits = model(images, training=False)
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32).numpy()
        predict_time_sec += time.perf_counter() - predict_start
        total_samples += batch_size
        labels_np = labels.numpy()
        variant_np = variant_types.numpy()
        update_confusion(state["ALL"], labels_np, predictions, num_classes=num_classes)
        for variant_type in (1, 2):
            mask = variant_np == variant_type
            if np.any(mask):
                update_confusion(
                    state[variant_type],
                    labels_np[mask],
                    predictions[mask],
                    num_classes=num_classes,
                )

    samples_per_second = total_samples / predict_time_sec if predict_time_sec else math.nan
    ms_per_sample = predict_time_sec * 1000 / total_samples if total_samples else math.nan
    params = int(model.count_params())
    params_million = params / 1_000_000
    checkpoint_mb = model_path.stat().st_size / (1024 * 1024)
    cost_score = ms_per_sample * params_million if not math.isnan(ms_per_sample) else math.nan

    speed = {
        "samples": total_samples,
        "load_time_sec": load_time_sec,
        "predict_time_sec": predict_time_sec,
        "samples_per_second": samples_per_second,
        "ms_per_sample": ms_per_sample,
        "params": params,
        "params_million": params_million,
        "checkpoint_mb": checkpoint_mb,
        "cost_score": cost_score,
    }
    return state, speed


def metrics_from_confusion(
    matrix: np.ndarray,
    *,
    model_name: str,
    group_name: str,
    class_names: list[str],
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    correct = np.diag(matrix)
    total = int(matrix.sum())
    accuracy = float(correct.sum() / total) if total else math.nan

    class_rows = []
    precisions = []
    recalls = []
    f1s = []
    for index, class_name in enumerate(class_names):
        precision = float(correct[index] / predicted[index]) if predicted[index] else 0.0
        recall = float(correct[index] / support[index]) if support[index] else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        class_rows.append(
            {
                "model": model_name,
                "group": group_name,
                "class": class_name,
                "support": int(support[index]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    macro_precision = float(np.mean(precisions)) if precisions else math.nan
    macro_recall = float(np.mean(recalls)) if recalls else math.nan
    macro_f1 = float(np.mean(f1s)) if f1s else math.nan
    weighted_f1 = (
        float(np.average(f1s, weights=support)) if int(support.sum()) and f1s else math.nan
    )
    return (
        {
            "model": model_name,
            "group": group_name,
            "support": total,
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        },
        class_rows,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.{digits}f}"
    return str(value)


def normalize(values: list[float], *, pad: float = 0.05) -> tuple[float, float]:
    finite = [value for value in values if not math.isnan(value)]
    if not finite:
        return 0.0, 1.0
    low = min(finite)
    high = max(finite)
    if low == high:
        low -= 0.5
        high += 0.5
    span = high - low
    return low - span * pad, high + span * pad


def svg_line_chart(
    title: str,
    series: dict[str, list[tuple[float, float]]],
    *,
    width: int = 980,
    height: int = 380,
    y_min: float | None = None,
    y_max: float | None = None,
) -> str:
    margin = {"left": 62, "right": 24, "top": 50, "bottom": 58}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    all_x = [x for points in series.values() for x, _ in points]
    all_y = [y for points in series.values() for _, y in points]
    x_min, x_max = normalize(all_x, pad=0.0)
    if x_min == x_max:
        x_max += 1
    calc_y_min, calc_y_max = normalize(all_y)
    y_min = calc_y_min if y_min is None else y_min
    y_max = calc_y_max if y_max is None else y_max

    def sx(x: float) -> float:
        return margin["left"] + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return margin["top"] + (y_max - y) / (y_max - y_min) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{width / 2}" y="28" text-anchor="middle" class="chart-title">{html.escape(title)}</text>',
        f'<rect x="{margin["left"]}" y="{margin["top"]}" width="{plot_w}" height="{plot_h}" class="plot-bg"/>',
    ]
    for i in range(6):
        y = margin["top"] + i * plot_h / 5
        value = y_max - i * (y_max - y_min) / 5
        parts.append(f'<line x1="{margin["left"]}" x2="{width - margin["right"]}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{margin["left"] - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis-label">{value:.3f}</text>')
    for i in range(6):
        x = margin["left"] + i * plot_w / 5
        value = x_min + i * (x_max - x_min) / 5
        parts.append(f'<text x="{x:.1f}" y="{height - 26}" text-anchor="middle" class="axis-label">{value:.0f}</text>')

    for index, (name, points) in enumerate(series.items()):
        color = PALETTE[index % len(PALETTE)]
        path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        legend_x = margin["left"] + index * 220
        legend_y = height - 8
        parts.append(f'<line x1="{legend_x}" x2="{legend_x + 24}" y1="{legend_y}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 30}" y="{legend_y + 4}" class="legend">{html.escape(name)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def svg_grouped_bar_chart(
    title: str,
    categories: list[str],
    series: dict[str, list[float]],
    *,
    width: int = 980,
    height: int = 420,
    y_min: float = 0.0,
    y_max: float | None = None,
) -> str:
    margin = {"left": 64, "right": 24, "top": 50, "bottom": 88}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    values = [value for vals in series.values() for value in vals]
    if y_max is None:
        _, y_max = normalize(values)
        y_max = max(y_max, 1.0 if max(values or [0]) <= 1.0 else max(values or [0]))
    group_w = plot_w / max(1, len(categories))
    names = list(series)
    bar_gap = 4
    bar_w = max(4, (group_w - 18) / max(1, len(names)) - bar_gap)

    def sy(value: float) -> float:
        return margin["top"] + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{width / 2}" y="28" text-anchor="middle" class="chart-title">{html.escape(title)}</text>',
        f'<rect x="{margin["left"]}" y="{margin["top"]}" width="{plot_w}" height="{plot_h}" class="plot-bg"/>',
    ]
    for i in range(6):
        y = margin["top"] + i * plot_h / 5
        value = y_max - i * (y_max - y_min) / 5
        parts.append(f'<line x1="{margin["left"]}" x2="{width - margin["right"]}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{margin["left"] - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis-label">{value:.3f}</text>')

    for cat_index, category in enumerate(categories):
        group_x = margin["left"] + cat_index * group_w
        label_x = group_x + group_w / 2
        parts.append(f'<text x="{label_x:.1f}" y="{height - 48}" text-anchor="middle" class="axis-label">{html.escape(category)}</text>')
        for series_index, name in enumerate(names):
            value = series[name][cat_index]
            color = PALETTE[series_index % len(PALETTE)]
            x = group_x + 9 + series_index * (bar_w + bar_gap)
            y = sy(value)
            h = margin["top"] + plot_h - y
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"><title>{html.escape(name)} {html.escape(category)}: {value:.4f}</title></rect>')

    for index, name in enumerate(names):
        legend_x = margin["left"] + index * 220
        legend_y = height - 16
        color = PALETTE[index % len(PALETTE)]
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="18" height="10" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 24}" y="{legend_y}" class="legend">{html.escape(name)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def svg_confusion_matrix(
    title: str,
    matrix: np.ndarray,
    class_names: list[str],
    *,
    cell: int = 62,
) -> str:
    size = cell * len(class_names)
    width = size + 170
    height = size + 132
    max_value = int(matrix.max()) if matrix.size else 0
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{width / 2}" y="24" text-anchor="middle" class="mini-title">{html.escape(title)}</text>',
        '<text x="18" y="92" class="axis-label" transform="rotate(-90 18,92)">True</text>',
        f'<text x="{95 + size / 2}" y="{height - 10}" text-anchor="middle" class="axis-label">Predicted</text>',
    ]
    for i, name in enumerate(class_names):
        parts.append(f'<text x="{95 + i * cell + cell / 2}" y="54" text-anchor="middle" class="axis-label">{html.escape(name)}</text>')
        parts.append(f'<text x="82" y="{76 + i * cell + cell / 2 + 4}" text-anchor="end" class="axis-label">{html.escape(name)}</text>')
    for r in range(len(class_names)):
        for c in range(len(class_names)):
            value = int(matrix[r, c])
            ratio = value / max_value if max_value else 0
            blue = int(245 - ratio * 150)
            fill = f"rgb({blue},{blue + 4},{255})"
            x = 95 + c * cell
            y = 76 + r * cell
            text_color = "#111827" if ratio < 0.55 else "#ffffff"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff"/>')
            parts.append(f'<text x="{x + cell / 2}" y="{y + cell / 2 + 5}" text-anchor="middle" fill="{text_color}" class="cell-label">{value}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def table_html(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(fmt(row.get(column, '')))}</td>" for column in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def dashboard_html(
    *,
    title: str,
    split: str,
    model_names: list[str],
    histories: dict[str, list[dict[str, float]]],
    history_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    speed_rows: list[dict[str, object]],
    class_rows: list[dict[str, object]],
    confusion: dict[str, dict[str, np.ndarray]],
    class_names: list[str],
) -> str:
    labels = [DEFAULT_MODEL_LABELS.get(name, name) for name in model_names]
    history_by_model = {row["model"]: row for row in history_rows}

    train_acc_series = {
        DEFAULT_MODEL_LABELS.get(name, name): [
            (row["epoch"] + 1, row["accuracy"]) for row in histories.get(name, [])
        ]
        for name in model_names
    }
    val_acc_series = {
        DEFAULT_MODEL_LABELS.get(name, name): [
            (row["epoch"] + 1, row["val_accuracy"]) for row in histories.get(name, [])
        ]
        for name in model_names
    }
    train_loss_series = {
        DEFAULT_MODEL_LABELS.get(name, name): [
            (row["epoch"] + 1, row["loss"]) for row in histories.get(name, [])
        ]
        for name in model_names
    }
    val_loss_series = {
        DEFAULT_MODEL_LABELS.get(name, name): [
            (row["epoch"] + 1, row["val_loss"]) for row in histories.get(name, [])
        ]
        for name in model_names
    }

    metric_by_group = defaultdict(dict)
    for row in metric_rows:
        metric_by_group[row["group"]][row["model"]] = row

    best_val_acc = {
        "best val accuracy": [
            float(history_by_model[name]["best_val_accuracy"]) for name in model_names
        ]
    }
    best_val_loss = {
        "best val loss": [
            float(history_by_model[name]["best_val_loss"]) for name in model_names
        ]
    }
    split_accuracy = {
        group: [float(metric_by_group[group][name]["accuracy"]) for name in model_names]
        for group in ("ALL", "SNP", "INDEL")
    }
    split_macro_f1 = {
        group: [float(metric_by_group[group][name]["macro_f1"]) for name in model_names]
        for group in ("ALL", "SNP", "INDEL")
    }
    speed_by_model = {row["model"]: row for row in speed_rows}
    throughput = {
        "samples/sec": [
            float(speed_by_model[name]["samples_per_second"]) for name in model_names
        ]
    }
    latency = {
        "ms/sample": [
            float(speed_by_model[name]["ms_per_sample"]) for name in model_names
        ]
    }
    cost_score = {
        "cost score": [
            float(speed_by_model[name]["cost_score"]) for name in model_names
        ]
    }
    params_million = {
        "params M": [
            float(speed_by_model[name]["params_million"]) for name in model_names
        ]
    }

    best_by_group = []
    for group in ("ALL", "SNP", "INDEL"):
        best = max(
            (metric_by_group[group][name] for name in model_names),
            key=lambda row: float(row["macro_f1"]),
        )
        best_by_group.append(
            {
                "group": group,
                "best_model": DEFAULT_MODEL_LABELS.get(str(best["model"]), str(best["model"])),
                "macro_f1": best["macro_f1"],
                "accuracy": best["accuracy"],
            }
        )
    fastest = min(speed_rows, key=lambda row: float(row["ms_per_sample"]))
    lightest = min(speed_rows, key=lambda row: float(row["cost_score"]))
    best_by_group.extend(
        [
            {
                "group": "FASTEST",
                "best_model": DEFAULT_MODEL_LABELS.get(str(fastest["model"]), str(fastest["model"])),
                "macro_f1": "",
                "accuracy": f'{float(fastest["samples_per_second"]):.2f} samples/sec',
            },
            {
                "group": "LOWEST_COST",
                "best_model": DEFAULT_MODEL_LABELS.get(str(lightest["model"]), str(lightest["model"])),
                "macro_f1": "",
                "accuracy": f'{float(lightest["cost_score"]):.4f} cost_score',
            },
        ]
    )

    confusion_svgs = []
    for model_name in model_names:
        model_label = DEFAULT_MODEL_LABELS.get(model_name, model_name)
        for group in ("SNP", "INDEL"):
            matrix = confusion[model_name][group]
            confusion_svgs.append(svg_confusion_matrix(f"{model_label} - {group}", matrix, class_names))

    style = """
    body { font-family: Arial, sans-serif; margin: 28px; color: #111827; background: #f8fafc; }
    h1, h2 { margin: 0 0 16px; }
    section { margin: 28px 0; padding: 20px; background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; }
    .grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 18px; }
    .grid4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 14px; }
    svg { width: 100%; height: auto; }
    .chart-title { font-weight: 700; font-size: 18px; }
    .mini-title { font-weight: 700; font-size: 14px; }
    .axis-label, .legend { font-size: 12px; fill: #374151; }
    .cell-label { font-size: 13px; font-weight: 700; }
    .plot-bg { fill: #ffffff; stroke: #d1d5db; }
    .grid { stroke: #e5e7eb; stroke-width: 1; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border: 1px solid #e5e7eb; padding: 7px 8px; text-align: right; }
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
    th { background: #f3f4f6; }
    code { background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }
    """

    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>{style}</style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Evaluation split: <code>{html.escape(split)}</code>. Variant type mapping: <code>1=SNP</code>, <code>2=INDEL</code>.</p>
  <section>
    <h2>Best Model By Group</h2>
    {table_html(best_by_group, ["group", "best_model", "macro_f1", "accuracy"])}
  </section>
  <section>
    <h2>Training Curves</h2>
    <div class="grid2">
      {svg_line_chart("Train Accuracy", train_acc_series, y_min=0.0, y_max=1.0)}
      {svg_line_chart("Validation Accuracy", val_acc_series, y_min=0.0, y_max=1.0)}
      {svg_line_chart("Train Loss", train_loss_series, y_min=0.0)}
      {svg_line_chart("Validation Loss", val_loss_series, y_min=0.0)}
    </div>
  </section>
  <section>
    <h2>Validation History Summary</h2>
    <div class="grid2">
      {svg_grouped_bar_chart("Best Validation Accuracy", labels, best_val_acc, y_min=0.0, y_max=1.0)}
      {svg_grouped_bar_chart("Best Validation Loss", labels, best_val_loss, y_min=0.0)}
    </div>
    {table_html(history_rows, ["model", "epochs", "best_val_accuracy", "best_val_accuracy_epoch", "best_val_loss", "best_val_loss_epoch", "last_val_accuracy", "last_val_loss"])}
  </section>
  <section>
    <h2>SNP vs INDEL Metrics</h2>
    <div class="grid2">
      {svg_grouped_bar_chart(f"{split.upper()} Accuracy By Variant Type", labels, split_accuracy, y_min=0.0, y_max=1.0)}
      {svg_grouped_bar_chart(f"{split.upper()} Macro F1 By Variant Type", labels, split_macro_f1, y_min=0.0, y_max=1.0)}
    </div>
    {table_html(metric_rows, ["model", "group", "support", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "samples_per_second", "ms_per_sample", "params_million", "checkpoint_mb", "cost_score"])}
  </section>
  <section>
    <h2>Speed And Cost</h2>
    <p><code>samples_per_second</code> cang cao cang nhanh. <code>ms_per_sample</code>, <code>params_million</code>, <code>checkpoint_mb</code>, va <code>cost_score = ms_per_sample * params_million</code> cang thap cang nhe.</p>
    <div class="grid2">
      {svg_grouped_bar_chart("Inference Throughput", labels, throughput, y_min=0.0)}
      {svg_grouped_bar_chart("Inference Latency", labels, latency, y_min=0.0)}
      {svg_grouped_bar_chart("Cost Score", labels, cost_score, y_min=0.0)}
      {svg_grouped_bar_chart("Model Size", labels, params_million, y_min=0.0)}
    </div>
    {table_html(speed_rows, ["model", "samples", "samples_per_second", "ms_per_sample", "predict_time_sec", "load_time_sec", "params_million", "checkpoint_mb", "cost_score", "speed_rank", "cost_rank"])}
  </section>
  <section>
    <h2>Per-Class Metrics</h2>
    {table_html(class_rows, ["model", "group", "class", "support", "precision", "recall", "f1"])}
  </section>
  <section>
    <h2>Confusion Matrices</h2>
    <div class="grid4">
      {''.join(confusion_svgs)}
    </div>
  </section>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "evaluation" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown_models = [name for name in model_names if name not in MODEL_BUILDERS]
    if unknown_models:
        raise ValueError(f"Unknown models: {', '.join(unknown_models)}")

    input_shape = (
        parse_input_shape(args.input_shape)
        if args.input_shape
        else infer_input_shape(args.data_dir, DEFAULT_TRAIN_PATTERN)
    )
    pattern = args.pattern or split_pattern(args.split)
    files = list_tfrecords(args.data_dir, pattern)
    class_names = [part.strip() for part in args.class_names.split(",") if part.strip()]
    if len(class_names) != args.num_classes:
        raise ValueError("--class-names length must match --num-classes")

    histories = {
        model_name: read_history(run_dir / model_name / "history.csv")
        for model_name in model_names
    }
    history_rows = [history_summary(model_name, histories[model_name]) for model_name in model_names]

    metric_rows: list[dict[str, object]] = []
    speed_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    confusion_by_model: dict[str, dict[str, np.ndarray]] = {}

    print(f"Evaluating split={args.split}, files={len(files)}, batch_size={args.batch_size}")
    for model_name in model_names:
        checkpoint_path = run_dir / model_name / f"{args.checkpoint}.keras"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        print(f"Evaluating {model_name}: {checkpoint_path}")
        dataset = make_eval_dataset(
            files,
            input_shape=input_shape,
            batch_size=args.batch_size,
            compression_type=args.compression_type,
            image_key=args.image_key,
            label_key=args.label_key,
            variant_type_key=args.variant_type_key,
            shape_key=args.shape_key,
            normalize=not args.no_normalize,
        )
        confusion_state, speed = evaluate_model(
            checkpoint_path,
            dataset,
            num_classes=args.num_classes,
        )
        speed_row = {"model": model_name, **speed}
        speed_rows.append(speed_row)
        confusion_by_model[model_name] = {
            "ALL": confusion_state["ALL"],
            "SNP": confusion_state[1],
            "INDEL": confusion_state[2],
        }

        for group_name, matrix in confusion_by_model[model_name].items():
            metric_row, per_class_rows = metrics_from_confusion(
                matrix,
                model_name=model_name,
                group_name=group_name,
                class_names=class_names,
            )
            metric_row.update(
                {
                    "samples_per_second": speed["samples_per_second"],
                    "ms_per_sample": speed["ms_per_sample"],
                    "params_million": speed["params_million"],
                    "checkpoint_mb": speed["checkpoint_mb"],
                    "cost_score": speed["cost_score"],
                }
            )
            metric_rows.append(metric_row)
            class_rows.extend(per_class_rows)
            for true_index, true_name in enumerate(class_names):
                for pred_index, pred_name in enumerate(class_names):
                    confusion_rows.append(
                        {
                            "model": model_name,
                            "group": group_name,
                            "true_class": true_name,
                            "predicted_class": pred_name,
                            "count": int(matrix[true_index, pred_index]),
                        }
                    )

    for rank, row in enumerate(
        sorted(speed_rows, key=lambda item: float(item["ms_per_sample"])),
        start=1,
    ):
        row["speed_rank"] = rank
    for rank, row in enumerate(
        sorted(speed_rows, key=lambda item: float(item["cost_score"])),
        start=1,
    ):
        row["cost_rank"] = rank
    rank_by_model = {
        row["model"]: {
            "speed_rank": row["speed_rank"],
            "cost_rank": row["cost_rank"],
        }
        for row in speed_rows
    }
    for row in metric_rows:
        row.update(rank_by_model[row["model"]])

    write_csv(output_dir / "history_summary.csv", history_rows)
    write_csv(output_dir / "split_metrics.csv", metric_rows)
    write_csv(output_dir / "speed_metrics.csv", speed_rows)
    write_csv(output_dir / "class_metrics.csv", class_rows)
    write_csv(output_dir / "confusion_matrices.csv", confusion_rows)

    metadata = {
        "run_dir": str(run_dir),
        "split": args.split,
        "pattern": pattern,
        "checkpoint": args.checkpoint,
        "models": model_names,
        "input_shape": list(input_shape),
        "batch_size": args.batch_size,
        "variant_type_names": DEFAULT_VARIANT_TYPE_NAMES,
        "speed_note": (
            "Prediction timing excludes TFRecord decode and model load. "
            "cost_score = ms_per_sample * params_million; lower is lighter/faster."
        ),
        "files": files,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    dashboard = dashboard_html(
        title=f"DeepVariant Model Evaluation - {run_dir.name}",
        split=args.split,
        model_names=model_names,
        histories=histories,
        history_rows=history_rows,
        metric_rows=metric_rows,
        speed_rows=speed_rows,
        class_rows=class_rows,
        confusion=confusion_by_model,
        class_names=class_names,
    )
    (output_dir / "dashboard.html").write_text(dashboard, encoding="utf-8")
    print(f"Wrote dashboard: {output_dir / 'dashboard.html'}")


if __name__ == "__main__":
    main()
