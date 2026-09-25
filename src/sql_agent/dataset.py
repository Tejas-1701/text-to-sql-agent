import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Example:
    question_id: int
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    difficulty: str


def find_questions_file(data_root: Path) -> Path:
    matches = sorted(data_root.rglob("mini_dev_sqlite.json"))
    if not matches:
        raise FileNotFoundError(f"mini_dev_sqlite.json not found under {data_root}")
    return matches[0]


def find_databases_dir(data_root: Path) -> Path:
    for candidate in sorted(data_root.rglob("dev_databases")):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"dev_databases folder not found under {data_root}")


def database_path(databases_dir: Path, db_id: str) -> Path:
    return databases_dir / db_id / f"{db_id}.sqlite"


def load_examples(data_root: Path) -> list[Example]:
    records = json.loads(find_questions_file(data_root).read_text())
    return [
        Example(
            question_id=int(record["question_id"]),
            db_id=record["db_id"],
            question=record["question"],
            evidence=record.get("evidence", "") or "",
            gold_sql=record["SQL"],
            difficulty=record.get("difficulty", "unknown"),
        )
        for record in records
    ]


def fixed_subset(examples: list[Example], size: int, seed: int = 42) -> list[Example]:
    if size >= len(examples):
        return list(examples)
    by_difficulty: dict[str, list[Example]] = {}
    for example in examples:
        by_difficulty.setdefault(example.difficulty, []).append(example)
    generator = random.Random(seed)
    chosen: list[Example] = []
    for difficulty in sorted(by_difficulty):
        group = by_difficulty[difficulty]
        share = round(size * len(group) / len(examples))
        chosen.extend(generator.sample(group, min(share, len(group))))
    remaining = [example for example in examples if example not in chosen]
    while len(chosen) < size:
        chosen.append(remaining.pop(generator.randrange(len(remaining))))
    return sorted(chosen[:size], key=lambda example: example.question_id)
