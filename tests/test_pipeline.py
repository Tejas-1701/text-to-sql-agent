import json
import sqlite3
from pathlib import Path

import pytest

from sql_agent.agent import BaselineAgent, extract_sql
from sql_agent.dataset import Example, fixed_subset, load_examples
from sql_agent.evaluate import markdown_table, run_evaluation, summarize
from sql_agent.executor import execute_sql, results_match
from sql_agent.llm import Completion
from sql_agent.schema import describe_database


class ScriptedModel:
    name = "scripted"

    def __init__(self, answers: list[str]):
        self.answers = list(answers)

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> Completion:
        return Completion(text=self.answers.pop(0), input_tokens=100, output_tokens=20, seconds=0.01)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    database_folder = tmp_path / "MINIDEV" / "dev_databases" / "school"
    database_folder.mkdir(parents=True)
    connection = sqlite3.connect(database_folder / "school.sqlite")
    connection.executescript(
        """
        CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, grade INTEGER);
        INSERT INTO students VALUES (1, 'Asha', 90), (2, 'Ben', 75), (3, 'Chen', 82);
        """
    )
    connection.commit()
    connection.close()
    questions = [
        {"question_id": 0, "db_id": "school", "question": "How many students?", "evidence": "",
         "SQL": "SELECT COUNT(*) FROM students", "difficulty": "simple"},
        {"question_id": 1, "db_id": "school", "question": "Who scored above 80?", "evidence": "above 80 means grade > 80",
         "SQL": "SELECT name FROM students WHERE grade > 80", "difficulty": "moderate"},
        {"question_id": 2, "db_id": "school", "question": "Top student?", "evidence": "",
         "SQL": "SELECT name FROM students ORDER BY grade DESC LIMIT 1", "difficulty": "challenging"},
    ]
    (tmp_path / "MINIDEV" / "mini_dev_sqlite.json").write_text(json.dumps(questions))
    return tmp_path


def test_extract_sql_variants():
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("Here you go: SELECT name FROM t;") == "SELECT name FROM t"
    assert extract_sql("```\nWITH a AS (SELECT 1) SELECT * FROM a\n```").startswith("WITH")
    assert extract_sql("``` sql\nSELECT 2\n```") == "SELECT 2"
    assert extract_sql("```\nsql\nSELECT 3\n```") == "SELECT 3"
    assert extract_sql("```SQL SELECT 4```") == "SELECT 4"
    assert extract_sql("```sql\nSELECT 5\n") == "SELECT 5"
    assert extract_sql("````sql\nSELECT 6\n````") == "SELECT 6"
    assert extract_sql("Query:\n```\n```sql\nSELECT 7\n```\n```") == "SELECT 7"
    assert extract_sql("Use `frpm`.\n```sql\nSELECT `School Name` FROM frpm;\n```") == "SELECT `School Name` FROM frpm"


def test_executor_is_read_only_and_reports_errors(data_root: Path):
    db_path = data_root / "MINIDEV" / "dev_databases" / "school" / "school.sqlite"
    assert execute_sql(db_path, "DELETE FROM students").error
    assert execute_sql(db_path, "SELECT nope FROM students").error
    assert execute_sql(db_path, "SELECT COUNT(*) FROM students").rows == [(3,)]


def test_executor_timeout(data_root: Path):
    db_path = data_root / "MINIDEV" / "dev_databases" / "school" / "school.sqlite"
    slow = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT COUNT(*) FROM n"
    result = execute_sql(db_path, slow, timeout_seconds=0.5)
    assert result.timed_out


def test_results_match_ignores_order(data_root: Path):
    db_path = data_root / "MINIDEV" / "dev_databases" / "school" / "school.sqlite"
    first = execute_sql(db_path, "SELECT name FROM students ORDER BY name")
    second = execute_sql(db_path, "SELECT name FROM students ORDER BY name DESC")
    assert results_match(first, second)


def test_schema_includes_sample_values(data_root: Path):
    db_path = data_root / "MINIDEV" / "dev_databases" / "school" / "school.sqlite"
    text = describe_database(db_path, 2)
    assert "CREATE TABLE students" in text
    assert "Asha" in text


def test_fixed_subset_is_stable():
    examples = [Example(i, "db", "q", "", "SELECT 1", ["simple", "moderate"][i % 2]) for i in range(50)]
    assert fixed_subset(examples, 10) == fixed_subset(examples, 10)
    assert len(fixed_subset(examples, 10)) == 10


def test_end_to_end_run_resumes_and_summarizes(data_root: Path, tmp_path: Path):
    examples = load_examples(data_root)
    databases_dir = data_root / "MINIDEV" / "dev_databases"
    output_path = tmp_path / "results" / "baseline__scripted__n3.jsonl"
    model = ScriptedModel(["```sql\nSELECT COUNT(*) FROM students\n```", "SELECT name FROM students WHERE grade > 80"])
    run_evaluation(BaselineAgent(model), examples[:2], databases_dir, output_path, progress=lambda message: None)
    model.answers = ["```sql\nSELECT bad_column FROM students\n```"]
    run_evaluation(BaselineAgent(model), examples, databases_dir, output_path, progress=lambda message: None)
    summary = summarize(output_path, input_price_per_million=1.0, output_price_per_million=4.0)
    assert summary["questions"] == 3
    assert summary["execution_accuracy"] == pytest.approx(2 / 3)
    assert summary["execution_error_rate"] == pytest.approx(1 / 3)
    assert summary["cost_per_query_usd"] == pytest.approx((100 * 1 + 20 * 4) / 1_000_000)
    assert "baseline__scripted__n3" in markdown_table([summary])


def test_network_drops_are_retried():
    import httpx
    from sql_agent.llm import is_retryable, with_retries

    assert is_retryable(httpx.RemoteProtocolError("Server disconnected without sending a response."))
    assert not is_retryable(ValueError("bad model name"))
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return "done"

    assert with_retries(flaky, first_delay=0.0) == "done"
    assert len(calls) == 3


def test_duplicate_question_ids_are_both_run(data_root: Path, tmp_path: Path):
    databases_dir = data_root / "MINIDEV" / "dev_databases"
    examples = [
        Example(137, "school", "How many students?", "", "SELECT COUNT(*) FROM students", "simple"),
        Example(137, "school", "Top student?", "", "SELECT name FROM students ORDER BY grade DESC LIMIT 1", "simple"),
    ]
    output_path = tmp_path / "duplicates.jsonl"
    model = ScriptedModel(["SELECT COUNT(*) FROM students"])
    run_evaluation(BaselineAgent(model), examples[:1], databases_dir, output_path, progress=lambda message: None)
    model.answers = ["SELECT name FROM students ORDER BY grade DESC LIMIT 1"]
    run_evaluation(BaselineAgent(model), examples, databases_dir, output_path, progress=lambda message: None)
    assert summarize(output_path)["questions"] == 2


def write_descriptions(data_root: Path) -> None:
    folder = data_root / "MINIDEV" / "dev_databases" / "school" / "database_description"
    folder.mkdir(exist_ok=True)
    text = (
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "id,,student id,integer,\n"
        "name,,full name of the student,text,\n"
        "grade,final grade,final exam grade,integer,\"0-100; above 80 is honours, café note\"\n"
    )
    (folder / "students.csv").write_bytes(text.encode("cp1252"))


def test_column_descriptions_read_bird_csvs(data_root: Path):
    from sql_agent.context import column_descriptions

    write_descriptions(data_root)
    db_path = data_root / "MINIDEV" / "dev_databases" / "school" / "school.sqlite"
    text = column_descriptions(db_path)
    assert "Table students:" in text
    assert "name: full name of the student" in text
    assert "values: 0-100; above 80 is honours, café note" in text


def test_value_matching_finds_question_values(data_root: Path):
    from sql_agent.context import candidate_phrases, matched_values

    db_path = data_root / "MINIDEV" / "dev_databases" / "school" / "school.sqlite"
    assert "the" not in candidate_phrases("What is the grade of Chen?")
    text = matched_values(db_path, "What grade did chen get?", "")
    assert "\"students\".\"name\" = 'Chen'" in text
    assert matched_values(db_path, "How many rows are there?", "") == ""


def test_variants_change_the_prompt(data_root: Path):
    from sql_agent.agent import build_agent

    write_descriptions(data_root)
    db_path = data_root / "MINIDEV" / "dev_databases" / "school" / "school.sqlite"
    question = "What grade did Asha get?"
    baseline = build_agent("baseline", ScriptedModel([])).build_prompt(db_path, question, "")
    full = build_agent("descriptions_value_match", ScriptedModel([])).build_prompt(db_path, question, "")
    assert "Column descriptions" not in baseline and "found in the database" not in baseline
    assert "Column descriptions" in full and "'Asha'" in full
    assert full.index("Column descriptions") < full.index("Question:")


def test_compare_counts_and_mcnemar(tmp_path: Path):
    from sql_agent.evaluate import compare_runs, mcnemar_exact_p

    assert round(mcnemar_exact_p(10, 14), 3) == 0.541
    assert mcnemar_exact_p(0, 0) == 1.0
    first = [(1, "a", True), (2, "b", True), (3, "c", False), (4, "d", False)]
    second = [(1, "a", True), (2, "b", False), (3, "c", True), (4, "d", True), (5, "e", True)]
    for name, rows in (("first.jsonl", first), ("second.jsonl", second)):
        lines = [json.dumps({"question_id": qid, "question": text, "correct": ok}) for qid, text, ok in rows]
        (tmp_path / name).write_text("\n".join(lines) + "\n")
    result = compare_runs(tmp_path / "first.jsonl", tmp_path / "second.jsonl")
    assert result["shared_questions"] == 4
    assert (result["fixed_by_second"], result["broken_by_second"], result["both_correct"]) == (2, 1, 1)


def test_descriptions_skip_ones_that_repeat_the_column_name(data_root: Path):
    from sql_agent.context import column_descriptions

    folder = data_root / "MINIDEV" / "dev_databases" / "school" / "database_description"
    folder.mkdir(exist_ok=True)
    (folder / "students.csv").write_text(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "GasStationID,,Gas Station ID,integer,\n"
        "grade,,final exam grade,integer,\n"
    )
    db_path = data_root / "MINIDEV" / "dev_databases" / "school" / "school.sqlite"
    column_descriptions.cache_clear()
    text = column_descriptions(db_path)
    assert "GasStationID" not in text
    assert "grade: final exam grade" in text


class RecordingModel(ScriptedModel):
    def __init__(self, answers: list[str]):
        super().__init__(answers)
        self.prompts: list[str] = []

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> Completion:
        self.prompts.append(user_prompt)
        return super().complete(system_prompt, user_prompt, temperature)


def school_db(data_root: Path) -> Path:
    return data_root / "MINIDEV" / "dev_databases" / "school" / "school.sqlite"


def test_self_correction_fixes_an_error(data_root: Path):
    model = RecordingModel(["SELECT nope FROM students", "SELECT COUNT(*) FROM students"])
    result = BaselineAgent(model, max_corrections=2).answer(school_db(data_root), "How many students?")
    assert result.llm_calls == 2
    assert result.execution.rows == [(3,)]
    assert result.first_execution.error
    assert "no such column: nope" in model.prompts[1]
    assert "SELECT nope FROM students" in model.prompts[1]


def test_self_correction_retries_empty_results(data_root: Path):
    model = RecordingModel(["SELECT name FROM students WHERE name = 'asha'", "SELECT name FROM students WHERE name = 'Asha'"])
    result = BaselineAgent(model, max_corrections=2).answer(school_db(data_root), "Is Asha a student?")
    assert result.llm_calls == 2
    assert result.execution.rows == [("Asha",)]
    assert "returned no rows" in model.prompts[1]


def test_self_correction_skips_good_answers_and_stops_when_unchanged(data_root: Path):
    good = RecordingModel(["SELECT COUNT(*) FROM students"])
    assert BaselineAgent(good, max_corrections=2).answer(school_db(data_root), "How many?").llm_calls == 1
    stubborn = RecordingModel(["SELECT name FROM students WHERE grade > 100", "SELECT name FROM students WHERE grade > 100"])
    result = BaselineAgent(stubborn, max_corrections=2).answer(school_db(data_root), "Who scored over 100?")
    assert result.llm_calls == 2
    assert result.execution.rows == []


def test_self_correction_keeps_a_runnable_query_over_a_broken_retry(data_root: Path):
    model = RecordingModel(["SELECT name FROM students WHERE grade > 100", "SELECT broken FROM", "SELECT also broken FROM"])
    result = BaselineAgent(model, max_corrections=2).answer(school_db(data_root), "Who scored over 100?")
    assert result.llm_calls == 3
    assert result.execution.error is None
    assert result.sql == "SELECT name FROM students WHERE grade > 100"


def test_baseline_never_retries(data_root: Path):
    model = RecordingModel(["SELECT nope FROM students"])
    result = BaselineAgent(model).answer(school_db(data_root), "How many students?")
    assert result.llm_calls == 1 and result.execution.error


def test_retry_metrics_in_summary(data_root: Path, tmp_path: Path):
    examples = load_examples(data_root)[:2]
    model = RecordingModel(["SELECT nope FROM students", "SELECT COUNT(*) FROM students", "SELECT name FROM students WHERE grade > 80"])
    output_path = tmp_path / "self_correct.jsonl"
    run_evaluation(BaselineAgent(model, max_corrections=2), examples, data_root / "MINIDEV" / "dev_databases", output_path, progress=lambda message: None)
    summary = summarize(output_path)
    assert summary["execution_accuracy"] == 1.0
    assert summary["questions_retried"] == 1
    assert summary["fixed_after_first_attempt"] == 1
    assert summary["broken_after_first_attempt"] == 0
    assert summary["first_attempt_accuracy"] == 0.5


def test_daily_quota_is_not_retried():
    from sql_agent.llm import is_retryable

    daily = RuntimeError("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    assert not is_retryable(daily)
    assert is_retryable(RuntimeError("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerMinutePerProjectPerModel"))


def test_same_id_and_text_with_different_sql_are_separate(data_root: Path, tmp_path: Path):
    databases_dir = data_root / "MINIDEV" / "dev_databases"
    examples = [
        Example(137, "school", "Same wording", "", "SELECT COUNT(*) FROM students", "simple"),
        Example(137, "school", "Same wording", "", "SELECT MAX(grade) FROM students", "simple"),
    ]
    output_path = tmp_path / "same.jsonl"
    model = ScriptedModel(["SELECT COUNT(*) FROM students", "SELECT MAX(grade) FROM students"])
    run_evaluation(BaselineAgent(model), examples, databases_dir, output_path, progress=lambda message: None)
    assert summarize(output_path)["questions"] == 2


class TemperatureRecordingModel(ScriptedModel):
    def __init__(self, answers: list[str]):
        super().__init__(answers)
        self.temperatures: list[float] = []

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> Completion:
        self.temperatures.append(temperature)
        return super().complete(system_prompt, user_prompt, temperature)


def test_vote_picks_the_majority_result(data_root: Path):
    model = TemperatureRecordingModel([
        "SELECT 5",
        "SELECT COUNT(*) FROM students",
        "SELECT COUNT(id) FROM students",
        "SELECT COUNT(name) FROM students",
        "SELECT 5",
    ])
    result = BaselineAgent(model, candidates=5).answer(school_db(data_root), "How many students?")
    assert result.execution.rows == [(3,)]
    assert result.sql == "SELECT COUNT(*) FROM students"
    assert result.note == "(3/5 agree)"
    assert result.llm_calls == 5
    assert model.temperatures == [0.0, 0.7, 0.7, 0.7, 0.7]
    assert result.first_execution.rows == [(5,)]


def test_vote_ignores_errors_and_prefers_non_empty_results(data_root: Path):
    model = ScriptedModel([
        "SELECT nope FROM students",
        "SELECT name FROM students WHERE grade > 100",
        "SELECT name FROM students WHERE grade > 100",
        "SELECT name FROM students WHERE grade > 85",
        "SELECT broken FROM",
    ])
    result = BaselineAgent(model, candidates=5).answer(school_db(data_root), "Who got over 85?")
    assert result.execution.rows == [("Asha",)]


def test_vote_ties_go_to_the_earliest_candidate_and_ignore_row_order(data_root: Path):
    model = ScriptedModel([
        "SELECT name FROM students ORDER BY name",
        "SELECT name FROM students ORDER BY name DESC",
        "SELECT 1",
        "SELECT 1",
        "SELECT name FROM students WHERE id = 1",
    ])
    result = BaselineAgent(model, candidates=5).answer(school_db(data_root), "List students")
    assert result.sql == "SELECT name FROM students ORDER BY name"
    assert result.note == "(2/5 agree)"


def test_vote_with_only_errors_falls_back_to_the_first_candidate(data_root: Path):
    model = ScriptedModel(["SELECT a FROM", "SELECT b FROM", "SELECT c FROM"])
    result = BaselineAgent(model, candidates=3).answer(school_db(data_root), "Anything")
    assert result.sql == "SELECT a FROM"
    assert result.execution.error


def test_vote_metrics_in_summary(data_root: Path, tmp_path: Path):
    examples = load_examples(data_root)[:1]
    model = ScriptedModel(["SELECT 7", "SELECT COUNT(*) FROM students", "SELECT COUNT(*) FROM students", "SELECT 9", "SELECT 10"])
    output_path = tmp_path / "vote.jsonl"
    run_evaluation(BaselineAgent(model, candidates=5), examples, data_root / "MINIDEV" / "dev_databases", output_path, progress=lambda message: None)
    summary = summarize(output_path)
    assert summary["execution_accuracy"] == 1.0
    assert summary["first_attempt_accuracy"] == 0.0
    assert summary["fixed_after_first_attempt"] == 1
    assert summary["any_candidate_correct"] == 1.0
    assert summary["avg_llm_calls"] == 5


def test_any_candidate_correct_counts_losing_candidates(data_root: Path, tmp_path: Path):
    examples = load_examples(data_root)[:1]
    model = ScriptedModel(["SELECT 7", "SELECT 7", "SELECT COUNT(*) FROM students"])
    output_path = tmp_path / "vote_lost.jsonl"
    run_evaluation(BaselineAgent(model, candidates=3), examples, data_root / "MINIDEV" / "dev_databases", output_path, progress=lambda message: None)
    summary = summarize(output_path)
    assert summary["execution_accuracy"] == 0.0
    assert summary["any_candidate_correct"] == 1.0
