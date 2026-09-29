import csv
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

from .dataset import database_path, load_examples
from .evaluate import example_key, read_records, record_key
from .executor import ExecutionResult, execute_sql


def preview(execution: ExecutionResult, limit: int = 3, width: int = 30) -> str:
    if execution.error:
        return f"ERROR: {execution.error}"
    lines = []
    for row in execution.rows[:limit]:
        cells = [str(value) if len(str(value)) <= width else str(value)[:width] + "..." for value in row]
        lines.append("(" + ", ".join(cells) + ")")
    if len(execution.rows) > limit:
        lines.append(f"... {len(execution.rows)} rows")
    return " | ".join(lines)


def matches_after_dropping_columns(predicted: ExecutionResult, gold: ExecutionResult, max_columns: int = 8) -> bool:
    predicted_width = len(predicted.columns)
    gold_width = len(gold.columns)
    if not predicted.rows or predicted_width <= gold_width or predicted_width > max_columns:
        return False
    target = set(gold.rows)
    for kept in combinations(range(predicted_width), gold_width):
        if {tuple(row[index] for index in kept) for row in predicted.rows} == target:
            return True
    return False


def numbers_close(predicted: ExecutionResult, gold: ExecutionResult, tolerance: float = 0.01) -> bool:
    if len(predicted.rows) != 1 or len(gold.rows) != 1 or len(predicted.rows[0]) != len(gold.rows[0]):
        return False
    for predicted_value, gold_value in zip(predicted.rows[0], gold.rows[0]):
        if not isinstance(predicted_value, (int, float)) or not isinstance(gold_value, (int, float)):
            return False
        scale = max(abs(gold_value), 1e-9)
        if abs(predicted_value - gold_value) / scale > tolerance:
            return False
    return True


def auto_flag(predicted: ExecutionResult, gold: ExecutionResult) -> str:
    if predicted.error:
        return "crashed"
    if not predicted.rows and gold.rows:
        return "empty result"
    if matches_after_dropping_columns(predicted, gold):
        return "right answer plus extra columns"
    if len(predicted.columns) > len(gold.columns):
        return "extra columns"
    if len(predicted.columns) < len(gold.columns):
        return "missing columns"
    if numbers_close(predicted, gold):
        return "number off by under 1%"
    if len(predicted.rows) > len(gold.rows):
        return "too many rows"
    if len(predicted.rows) < len(gold.rows):
        return "too few rows"
    return "same shape, different values"


def export_errors(
    result_paths: list[Path],
    data_root: Path,
    databases_dir: Path,
    output_path: Path,
    sample_size: int = 30,
    seed: int = 42,
) -> Counter:
    runs = [{record_key(record): record for record in read_records(path)} for path in result_paths]
    shared = set.intersection(*(set(run) for run in runs))
    always_wrong = sorted(key for key in shared if not any(run[key]["correct"] for run in runs))
    evidence_by_key = {
        example_key(example.question_id, example.question, example.gold_sql): example.evidence
        for example in load_examples(data_root)
    }
    sampled = set(random.Random(seed).sample(always_wrong, min(sample_size, len(always_wrong))))
    flags: Counter = Counter()
    rows = []
    for key in always_wrong:
        record = runs[0][key]
        db_path = database_path(databases_dir, record["db_id"])
        predicted = execute_sql(db_path, record["predicted_sql"])
        gold = execute_sql(db_path, record["gold_sql"])
        flag = auto_flag(predicted, gold)
        flags[flag] += 1
        rows.append({
            "in_sample": "yes" if key in sampled else "",
            "question_id": record["question_id"],
            "db_id": record["db_id"],
            "difficulty": record["difficulty"],
            "question": record["question"],
            "hint": evidence_by_key.get(key, ""),
            "auto_flag": flag,
            "my_category": "",
            "notes": "",
            "predicted_sql": record["predicted_sql"],
            "gold_sql": record["gold_sql"],
            "predicted_columns": ", ".join(predicted.columns),
            "gold_columns": ", ".join(gold.columns),
            "predicted_row_count": len(predicted.rows),
            "gold_row_count": len(gold.rows),
            "predicted_preview": preview(predicted),
            "gold_preview": preview(gold),
        })
    rows.sort(key=lambda row: (row["in_sample"] != "yes", row["auto_flag"], row["question_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        fieldnames = list(rows[0]) if rows else ["question_id"]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return flags
