import re
from dataclasses import dataclass, field
from pathlib import Path

from .executor import ExecutionResult, execute_sql
from .llm import LanguageModel
from .context import column_descriptions, matched_values
from .schema import describe_database

SYSTEM_PROMPT = (
    "You are an expert SQLite analyst. Write one SQLite query that answers the question. "
    "Use only tables and columns from the schema. Return only the SQL inside a ```sql code block."
)


CORRECTION_INSTRUCTIONS = (
    "Your previous query may be wrong. Read the problem below, then write a corrected SQLite query. "
    "If you are confident the previous query already answers the question, return it unchanged. "
    "Return only the SQL inside a ```sql code block."
)


@dataclass
class AgentResult:
    sql: str
    execution: ExecutionResult
    first_execution: ExecutionResult | None = None
    candidate_executions: list[ExecutionResult] = field(default_factory=list)
    note: str = ""
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_seconds: float = 0.0
    attempts: list[dict] = field(default_factory=list)


def extract_sql(text: str) -> str:
    query_start = re.compile(r"\b(WITH|SELECT)\b", flags=re.IGNORECASE)
    fenced = re.findall(r"```(.*?)```", text, flags=re.DOTALL)
    blocks_with_query = [block for block in fenced if query_start.search(block)]
    candidate = blocks_with_query[-1] if blocks_with_query else text.replace("```", "")
    match = query_start.search(candidate)
    if match:
        candidate = candidate[match.start():]
    return candidate.strip().rstrip(";").strip()


def format_preview(rows: list[tuple], limit: int = 5, width: int = 40) -> str:
    lines = []
    for row in rows[:limit]:
        cells = [str(value) if len(str(value)) <= width else str(value)[:width] + "..." for value in row]
        lines.append("  (" + ", ".join(cells) + ")")
    return "\n".join(lines)


def find_problem(execution: ExecutionResult, max_rows: int = 500) -> str | None:
    if execution.timed_out:
        return f"The query was stopped: {execution.error}. It is probably missing a join condition or filter."
    if execution.error:
        return f"The query failed with this error: {execution.error}"
    if not execution.rows:
        return (
            "The query ran but returned no rows. Check the join conditions, the filter values "
            "(spelling, letter case, date format), and that each column belongs to the table you used."
        )
    if all(value is None for row in execution.rows for value in row):
        return "The query returned only NULL values. Check the columns, the joins, and whether a filter removed all matches."
    if len(execution.rows) > max_rows:
        return (
            f"The query returned {len(execution.rows)} rows. If the question asks for a single value, "
            "a count, or a top result, you may be missing an aggregation, a GROUP BY, or a LIMIT."
        )
    return None


def build_correction_prompt(original_prompt: str, sql: str, problem: str, execution: ExecutionResult) -> str:
    parts = [original_prompt, f"Previous query:\n```sql\n{sql}\n```", f"Problem: {problem}"]
    if execution.rows:
        parts.append(f"First rows returned:\n{format_preview(execution.rows)}")
    parts.append(CORRECTION_INSTRUCTIONS)
    return "\n\n".join(parts)


def choose_final(attempts: list[tuple[str, ExecutionResult, str | None]]) -> tuple[str, ExecutionResult]:
    for sql, execution, problem in reversed(attempts):
        if problem is None:
            return sql, execution
    for sql, execution, problem in reversed(attempts):
        if execution.error is None:
            return sql, execution
    sql, execution, _ = attempts[-1]
    return sql, execution


def result_signature(execution: ExecutionResult) -> frozenset:
    return frozenset(execution.rows)


def pick_by_vote(candidates: list[tuple[str, ExecutionResult]]) -> tuple[int, int]:
    runnable = [index for index, (_, execution) in enumerate(candidates) if execution.ok]
    if not runnable:
        return 0, 0
    non_empty = [index for index in runnable if candidates[index][1].rows]
    pool = non_empty or runnable
    groups: dict[frozenset, list[int]] = {}
    for index in pool:
        groups.setdefault(result_signature(candidates[index][1]), []).append(index)
    winning_group = max(groups.values(), key=lambda members: (len(members), -members[0]))
    return winning_group[0], len(winning_group)


def build_user_prompt(schema_text: str, question: str, evidence: str, extra_sections: list[str] | None = None) -> str:
    parts = [f"Database schema:\n{schema_text}"]
    parts.extend(section for section in (extra_sections or []) if section)
    parts.append(f"Question: {question}")
    if evidence:
        parts.append(f"Hint: {evidence}")
    return "\n\n".join(parts)


class BaselineAgent:
    name = "baseline"

    def __init__(
        self,
        model: LanguageModel,
        sample_values: int = 0,
        use_descriptions: bool = False,
        use_value_matching: bool = False,
        max_corrections: int = 0,
        candidates: int = 1,
        sampling_temperature: float = 0.7,
        timeout_seconds: float = 30.0,
    ):
        self.model = model
        self.sample_values = sample_values
        self.use_descriptions = use_descriptions
        self.use_value_matching = use_value_matching
        self.max_corrections = max_corrections
        self.candidates = candidates
        self.sampling_temperature = sampling_temperature
        self.timeout_seconds = timeout_seconds

    def build_prompt(self, db_path: Path, question: str, evidence: str) -> str:
        schema_text = describe_database(db_path, self.sample_values)
        extra_sections = []
        if self.use_descriptions:
            descriptions = column_descriptions(db_path)
            if descriptions:
                extra_sections.append(f"Column descriptions:\n{descriptions}")
        if self.use_value_matching:
            extra_sections.append(matched_values(db_path, question, evidence))
        return build_user_prompt(schema_text, question, evidence, extra_sections)

    def answer(self, db_path: Path, question: str, evidence: str = "") -> AgentResult:
        if self.candidates > 1:
            return self.answer_by_vote(db_path, question, evidence)
        original_prompt = self.build_prompt(db_path, question, evidence)
        prompt = original_prompt
        attempts: list[tuple[str, ExecutionResult, str | None]] = []
        log: list[dict] = []
        totals = {"calls": 0, "input": 0, "output": 0, "seconds": 0.0}
        for attempt_number in range(self.max_corrections + 1):
            completion = self.model.complete(SYSTEM_PROMPT, prompt)
            totals["calls"] += 1
            totals["input"] += completion.input_tokens
            totals["output"] += completion.output_tokens
            totals["seconds"] += completion.seconds
            sql = extract_sql(completion.text)
            execution = execute_sql(db_path, sql, self.timeout_seconds)
            problem = find_problem(execution) if self.max_corrections else None
            attempts.append((sql, execution, problem))
            log.append({"raw_response": completion.text, "sql": sql, "error": execution.error,
                        "rows": len(execution.rows), "problem": problem})
            unchanged = attempt_number > 0 and sql.strip() == attempts[-2][0].strip()
            if problem is None or unchanged or attempt_number == self.max_corrections:
                break
            prompt = build_correction_prompt(original_prompt, sql, problem, execution)
        final_sql, final_execution = choose_final(attempts)
        retries = totals["calls"] - 1
        return AgentResult(
            sql=final_sql,
            execution=final_execution,
            first_execution=attempts[0][1],
            candidate_executions=[execution for _, execution, _ in attempts],
            llm_calls=totals["calls"],
            input_tokens=totals["input"],
            output_tokens=totals["output"],
            llm_seconds=totals["seconds"],
            attempts=log,
            note=f"(+{retries} retry)" if retries else "",
        )

    def answer_by_vote(self, db_path: Path, question: str, evidence: str) -> AgentResult:
        prompt = self.build_prompt(db_path, question, evidence)
        candidates: list[tuple[str, ExecutionResult]] = []
        log: list[dict] = []
        totals = {"input": 0, "output": 0, "seconds": 0.0}
        for index in range(self.candidates):
            temperature = 0.0 if index == 0 else self.sampling_temperature
            completion = self.model.complete(SYSTEM_PROMPT, prompt, temperature=temperature)
            totals["input"] += completion.input_tokens
            totals["output"] += completion.output_tokens
            totals["seconds"] += completion.seconds
            sql = extract_sql(completion.text)
            execution = execute_sql(db_path, sql, self.timeout_seconds)
            candidates.append((sql, execution))
            log.append({"raw_response": completion.text, "sql": sql, "temperature": temperature,
                        "error": execution.error, "rows": len(execution.rows)})
        winner, votes = pick_by_vote(candidates)
        for index, entry in enumerate(log):
            entry["chosen"] = index == winner
        return AgentResult(
            sql=candidates[winner][0],
            execution=candidates[winner][1],
            first_execution=candidates[0][1],
            candidate_executions=[execution for _, execution in candidates],
            llm_calls=self.candidates,
            input_tokens=totals["input"],
            output_tokens=totals["output"],
            llm_seconds=totals["seconds"],
            attempts=log,
            note=f"({votes}/{self.candidates} agree)",
        )


VARIANTS = {
    "baseline": {},
    "schema_values": {"sample_values": 3},
    "descriptions": {"use_descriptions": True},
    "value_match": {"use_value_matching": True},
    "descriptions_value_match": {"use_descriptions": True, "use_value_matching": True},
    "self_correct": {"use_descriptions": True, "max_corrections": 2},
    "vote": {"use_descriptions": True, "candidates": 5},
}


def build_agent(variant: str, model: LanguageModel) -> BaselineAgent:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Choose from: {', '.join(VARIANTS)}")
    agent = BaselineAgent(model, **VARIANTS[variant])
    agent.name = variant
    return agent
