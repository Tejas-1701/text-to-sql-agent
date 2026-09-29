import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExecutionResult:
    rows: list[tuple] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    error: str | None = None
    timed_out: bool = False
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


def execute_sql(db_path: Path, sql: str, timeout_seconds: float = 30.0, max_rows: int = 10000) -> ExecutionResult:
    started = time.perf_counter()
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    connection.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
    deadline = started + timeout_seconds
    connection.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10000)
    try:
        cursor = connection.execute(sql)
        rows = cursor.fetchmany(max_rows)
        columns = [description[0] for description in cursor.description or []]
        return ExecutionResult(rows=rows, columns=columns, seconds=time.perf_counter() - started)
    except sqlite3.OperationalError as error:
        timed_out = "interrupted" in str(error).lower()
        message = f"query exceeded {timeout_seconds:.0f}s timeout" if timed_out else str(error)
        return ExecutionResult(error=message, timed_out=timed_out, seconds=time.perf_counter() - started)
    except sqlite3.Error as error:
        return ExecutionResult(error=str(error), seconds=time.perf_counter() - started)
    except Exception as error:
        return ExecutionResult(error=f"{type(error).__name__}: {error}", seconds=time.perf_counter() - started)
    finally:
        connection.close()


def results_match(predicted: ExecutionResult, gold: ExecutionResult) -> bool:
    if not predicted.ok or not gold.ok:
        return False
    return set(predicted.rows) == set(gold.rows)
