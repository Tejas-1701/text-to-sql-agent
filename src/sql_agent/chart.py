import math
from dataclasses import dataclass
from pathlib import Path

from .evaluate import read_records

VARIANT_LABELS = {
    "baseline": "Baseline",
    "schema_values": "+ sample values",
    "descriptions": "+ column descriptions",
    "value_match": "+ matched values",
    "descriptions_value_match": "+ descriptions and matched values",
    "self_correct": "+ descriptions and self-correction",
    "vote": "+ descriptions and 5-way voting",
    "hint_rules": "+ hint rules and matched values",
}

MODEL_LABELS = {
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "qwen2.5-coder-7b": "Qwen 2.5 Coder 7B (local)",
}

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


@dataclass
class RunPoint:
    model: str
    variant: str
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total

    @property
    def label(self) -> str:
        return VARIANT_LABELS.get(self.variant, self.variant)


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = correct / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return centre - margin, centre + margin


def parse_run_name(path: Path) -> tuple[str, str, int] | None:
    parts = path.stem.split("__")
    if len(parts) != 3 or not parts[2].startswith("n"):
        return None
    variant, model, size = parts
    try:
        return variant, model, int(size[1:])
    except ValueError:
        return None


def collect_points(results_dir: Path, subset_size: int = 200) -> list[RunPoint]:
    points = []
    for path in sorted(results_dir.glob("*.jsonl")):
        parsed = parse_run_name(path)
        if parsed is None or parsed[2] != subset_size:
            continue
        variant, model, _ = parsed
        records = read_records(path)
        if len(records) < subset_size:
            continue
        points.append(RunPoint(model, variant, sum(record["correct"] for record in records), len(records)))
    model_order = {model: index for index, model in enumerate(MODEL_LABELS)}
    variant_order = {variant: index for index, variant in enumerate(VARIANT_LABELS)}
    points.sort(key=lambda point: (model_order.get(point.model, 99), point.model, variant_order.get(point.variant, 99)))
    return points


def draw_chart(points: list[RunPoint], output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not points:
        raise ValueError("No complete runs found to chart")
    models = list(dict.fromkeys(point.model for point in points))
    colors = {model: SERIES_COLORS[index % len(SERIES_COLORS)] for index, model in enumerate(models)}
    reference = next((point for point in points if point.variant == "baseline"), points[0])
    reference_low, reference_high = wilson_interval(reference.correct, reference.total)

    positions = []
    current = 0.0
    for index, point in enumerate(points):
        if index and point.model != points[index - 1].model:
            current += 0.7
        positions.append(-current)
        current += 1.0
    height = 1.3 + 0.42 * (current + 0.5)
    figure, axis = plt.subplots(figsize=(9, height), dpi=160)
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)
    axis.axvspan(reference_low * 100, reference_high * 100, color=colors[reference.model], alpha=0.08, linewidth=0)
    axis.axvline(reference.accuracy * 100, color=colors[reference.model], linewidth=1, linestyle=(0, (3, 3)), alpha=0.6)

    for position, point in zip(positions, points):
        low, high = wilson_interval(point.correct, point.total)
        color = colors[point.model]
        axis.plot([low * 100, high * 100], [position, position], color=color, linewidth=2, solid_capstyle="round")
        axis.scatter(point.accuracy * 100, position, s=64, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
        axis.text(high * 100 + 0.8, position, f"{point.accuracy:.1%}", va="center", ha="left",
                  fontsize=9, color=TEXT_PRIMARY)

    axis.set_yticks(positions)
    axis.set_yticklabels([point.label for point in points], fontsize=9, color=TEXT_PRIMARY)
    axis.set_xlabel("Execution accuracy (%), with 95% confidence interval", fontsize=9, color=TEXT_SECONDARY)
    lows = [wilson_interval(point.correct, point.total)[0] * 100 for point in points]
    highs = [wilson_interval(point.correct, point.total)[1] * 100 for point in points]
    axis.set_xlim(math.floor(min(lows) / 5) * 5, math.ceil(max(highs) / 5) * 5 + 5)
    axis.tick_params(axis="x", colors=TEXT_SECONDARY, labelsize=8, length=0)
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)

    handles = [
        plt.Line2D([], [], marker="o", color=colors[model], markersize=7, linewidth=2, label=MODEL_LABELS.get(model, model))
        for model in models
    ]
    legend = axis.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(models),
                         frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)
    axis.set_title("No prompting technique beat the baseline; changing the model did",
                   fontsize=11, color=TEXT_PRIMARY, loc="left", pad=28)
    figure.text(0.02, 0.005,
                f"BIRD mini-dev, {reference.total} questions. Shaded band: baseline 95% interval. Intervals are per run; "
                "differences were tested question by question (McNemar).",
                fontsize=7.5, color=TEXT_SECONDARY, ha="left", va="bottom")
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, facecolor=SURFACE)
    plt.close(figure)
    return output_path
