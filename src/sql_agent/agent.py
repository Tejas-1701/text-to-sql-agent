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


@dataclass
class AgentResult:
    sql: str
    execution: ExecutionResult
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
        timeout_seconds: float = 30.0,
    ):
        self.model = model
        self.sample_values = sample_values
        self.use_descriptions = use_descriptions
        self.use_value_matching = use_value_matching
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
        completion = self.model.complete(SYSTEM_PROMPT, self.build_prompt(db_path, question, evidence))
        sql = extract_sql(completion.text)
        execution = execute_sql(db_path, sql, self.timeout_seconds)
        return AgentResult(
            sql=sql,
            execution=execution,
            llm_calls=1,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            llm_seconds=completion.seconds,
            attempts=[{"raw_response": completion.text, "sql": sql, "error": execution.error}],
        )


VARIANTS = {
    "baseline": {},
    "schema_values": {"sample_values": 3},
    "descriptions": {"use_descriptions": True},
    "value_match": {"use_value_matching": True},
    "descriptions_value_match": {"use_descriptions": True, "use_value_matching": True},
}


def build_agent(variant: str, model: LanguageModel) -> BaselineAgent:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Choose from: {', '.join(VARIANTS)}")
    agent = BaselineAgent(model, **VARIANTS[variant])
    agent.name = variant
    return agent
