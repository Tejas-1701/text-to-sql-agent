import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .dataset import Example, database_path
from .executor import execute_sql, results_match


@dataclass
class Record:
    question_id: int
    db_id: str
    difficulty: str
    question: str
    predicted_sql: str
    gold_sql: str
    correct: bool
    first_attempt_correct: bool
    error: str | None
    gold_error: str | None
    llm_calls: int
    input_tokens: int
    output_tokens: int
    llm_seconds: float
    total_seconds: float
    attempts: list


def example_key(question_id: int, question: str, gold_sql: str) -> tuple[int, str, str]:
    return (question_id, question, gold_sql.strip())


def record_key(record: dict) -> tuple[int, str, str]:
    return example_key(record["question_id"], record["question"], record.get("gold_sql", ""))


def read_records(output_path: Path) -> list[dict]:
    return [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]


def load_done_keys(output_path: Path) -> set[tuple[int, str, str]]:
    if not output_path.exists():
        return set()
    return {record_key(record) for record in read_records(output_path)}


def run_evaluation(agent, examples: list[Example], databases_dir: Path, output_path: Path, progress=print) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = load_done_keys(output_path)
    pending = [
        example for example in examples
        if example_key(example.question_id, example.question, example.gold_sql) not in done_keys
    ]
    progress(f"{len(examples) - len(pending)} already done, {len(pending)} to run -> {output_path}")
    with output_path.open("a") as output_file:
        for position, example in enumerate(pending, start=1):
            db_path = database_path(databases_dir, example.db_id)
            started = time.perf_counter()
            try:
                result = agent.answer(db_path, example.question, example.evidence)
            except Exception as error:
                progress(f"[{position}/{len(pending)}] q{example.question_id} stopped: {error}")
                raise
            gold = execute_sql(db_path, example.gold_sql)
            first_execution = result.first_execution or result.execution
            record = Record(
                question_id=example.question_id,
                db_id=example.db_id,
                difficulty=example.difficulty,
                question=example.question,
                predicted_sql=result.sql,
                gold_sql=example.gold_sql,
                correct=results_match(result.execution, gold),
                first_attempt_correct=results_match(first_execution, gold),
                error=result.execution.error,
                gold_error=gold.error,
                llm_calls=result.llm_calls,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                llm_seconds=round(result.llm_seconds, 3),
                total_seconds=round(time.perf_counter() - started, 3),
                attempts=result.attempts,
            )
            output_file.write(json.dumps(asdict(record)) + "\n")
            output_file.flush()
            mark = "ok " if record.correct else "err" if record.error else "no "
            retries = f"  (+{record.llm_calls - 1} retry)" if record.llm_calls > 1 else ""
            progress(f"[{position}/{len(pending)}] {mark} q{example.question_id} {example.db_id}{retries}")


def summarize(output_path: Path, input_price_per_million: float = 0.0, output_price_per_million: float = 0.0) -> dict:
    records = read_records(output_path)
    if not records:
        return {"file": output_path.name, "questions": 0}
    count = len(records)
    by_difficulty = defaultdict(list)
    for record in records:
        by_difficulty[record["difficulty"]].append(record["correct"])
    cost = sum(
        record["input_tokens"] * input_price_per_million + record["output_tokens"] * output_price_per_million
        for record in records
    ) / 1_000_000
    return {
        "file": output_path.name,
        "questions": count,
        "execution_accuracy": sum(record["correct"] for record in records) / count,
        "accuracy_by_difficulty": {level: sum(values) / len(values) for level, values in sorted(by_difficulty.items())},
        "execution_error_rate": sum(record["error"] is not None for record in records) / count,
        "gold_errors": sum(record["gold_error"] is not None for record in records),
        "avg_llm_calls": sum(record["llm_calls"] for record in records) / count,
        "avg_input_tokens": sum(record["input_tokens"] for record in records) / count,
        "avg_output_tokens": sum(record["output_tokens"] for record in records) / count,
        "cost_per_query_usd": cost / count,
        "avg_latency_seconds": sum(record["total_seconds"] for record in records) / count,
        "questions_retried": sum(record["llm_calls"] > 1 for record in records),
        "fixed_by_retry": sum(record["correct"] and not record.get("first_attempt_correct", record["correct"]) for record in records),
        "broken_by_retry": sum(record.get("first_attempt_correct", record["correct"]) and not record["correct"] for record in records),
    }


def markdown_table(summaries: list[dict]) -> str:
    header = "| Run | Questions | Exec. accuracy | Exec. errors | Avg LLM calls | Cost/query | Latency |"
    divider = "|---|---|---|---|---|---|---|"
    rows = [
        f"| {summary['file'].removesuffix('.jsonl')} | {summary['questions']} | {summary['execution_accuracy']:.1%} "
        f"| {summary['execution_error_rate']:.1%} | {summary['avg_llm_calls']:.2f} "
        f"| ${summary['cost_per_query_usd']:.5f} | {summary['avg_latency_seconds']:.2f}s |"
        for summary in summaries
        if summary["questions"]
    ]
    return "\n".join([header, divider, *rows])


def load_outcomes(output_path: Path) -> dict[tuple[int, str, str], bool]:
    return {record_key(record): record["correct"] for record in read_records(output_path)}


def mcnemar_exact_p(only_first: int, only_second: int) -> float:
    from math import comb

    discordant = only_first + only_second
    if discordant == 0:
        return 1.0
    smaller = min(only_first, only_second)
    tail = sum(comb(discordant, count) for count in range(smaller + 1)) / 2 ** discordant
    return min(1.0, 2 * tail)


def compare_runs(first_path: Path, second_path: Path) -> dict:
    first = load_outcomes(first_path)
    second = load_outcomes(second_path)
    shared = sorted(set(first) & set(second))
    only_first = sum(first[key] and not second[key] for key in shared)
    only_second = sum(second[key] and not first[key] for key in shared)
    return {
        "first": first_path.name.removesuffix(".jsonl"),
        "second": second_path.name.removesuffix(".jsonl"),
        "shared_questions": len(shared),
        "first_accuracy": sum(first[key] for key in shared) / len(shared) if shared else 0.0,
        "second_accuracy": sum(second[key] for key in shared) / len(shared) if shared else 0.0,
        "both_correct": sum(first[key] and second[key] for key in shared),
        "fixed_by_second": only_second,
        "broken_by_second": only_first,
        "both_wrong": sum(not first[key] and not second[key] for key in shared),
        "mcnemar_p": mcnemar_exact_p(only_first, only_second),
    }
