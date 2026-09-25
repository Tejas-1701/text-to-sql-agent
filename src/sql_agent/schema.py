import sqlite3
from functools import lru_cache
from pathlib import Path


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def shorten(value, limit: int = 40) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


@lru_cache(maxsize=64)
def describe_database(db_path: Path, sample_values: int = 3) -> str:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
    try:
        table_rows = connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        blocks = []
        for table_name, create_sql in table_rows:
            lines = [create_sql.strip() + ";"]
            if sample_values > 0:
                columns = [row[1] for row in connection.execute(f"PRAGMA table_info({quote_identifier(table_name)})")]
                examples = []
                for column in columns:
                    query = (
                        f"SELECT DISTINCT {quote_identifier(column)} FROM {quote_identifier(table_name)} "
                        f"WHERE {quote_identifier(column)} IS NOT NULL LIMIT {sample_values}"
                    )
                    try:
                        values = [shorten(row[0]) for row in connection.execute(query)]
                    except sqlite3.Error:
                        values = []
                    if values:
                        examples.append(f"  {column}: {', '.join(values)}")
                if examples:
                    lines.append(f"/* Example values from {table_name}:")
                    lines.extend(examples)
                    lines.append("*/")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
    finally:
        connection.close()
