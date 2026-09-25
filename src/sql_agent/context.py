import csv
import io
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

from .schema import quote_identifier

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "what", "which", "who", "whom", "whose", "how", "many",
    "much", "list", "name", "names", "give", "show", "find", "please", "are", "was", "were", "has", "have", "had",
    "does", "did", "all", "any", "each", "per", "among", "between", "than", "more", "less", "most", "least",
    "number", "total", "average", "count", "amount", "percentage", "ratio", "their", "they", "them", "its", "his",
    "her", "into", "out", "over", "under", "about", "after", "before", "during", "where", "when", "why", "not",
    "only", "also", "been", "being", "there", "these", "those", "such", "same", "other", "state", "indicate",
    "calculate", "refers", "refer", "mean", "means", "value", "values", "id",
}


def read_text_any_encoding(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def clean(text: str | None, limit: int) -> str:
    collapsed = re.sub(r"\s+", " ", (text or "").strip())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "..."


@lru_cache(maxsize=64)
def column_descriptions(db_path: Path, value_limit: int = 200) -> str:
    folder = db_path.parent / "database_description"
    if not folder.is_dir():
        return ""
    blocks = []
    for csv_path in sorted(folder.glob("*.csv")):
        rows = csv.DictReader(io.StringIO(read_text_any_encoding(csv_path)))
        lines = []
        for row in rows:
            row = {(key or "").strip().lower(): value for key, value in row.items()}
            column = clean(row.get("original_column_name"), 80)
            if not column:
                continue
            parts = []
            meaning = clean(row.get("column_description") or row.get("column_name"), 150)
            if meaning and squash(meaning) != squash(column):
                parts.append(meaning)
            values = clean(row.get("value_description"), value_limit)
            if values and values.lower() not in {"nan", "none", ""}:
                parts.append(f"values: {values}")
            if parts:
                lines.append(f"  {column}: {'; '.join(parts)}")
        if lines:
            blocks.append(f"Table {csv_path.stem}:\n" + "\n".join(lines))
    return "\n".join(blocks)


def is_text_column(declared_type: str) -> bool:
    upper = (declared_type or "").upper()
    return upper == "" or any(marker in upper for marker in ("CHAR", "TEXT", "CLOB", "VARCHAR"))


@lru_cache(maxsize=16)
def value_index(db_path: Path, max_values_per_column: int = 20000, max_length: int = 60) -> dict[str, set[tuple[str, str, str]]]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
    index: dict[str, set[tuple[str, str, str]]] = {}
    try:
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )]
        for table in tables:
            for column_info in connection.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall():
                column, declared_type = column_info[1], column_info[2]
                if not is_text_column(declared_type):
                    continue
                query = (
                    f"SELECT DISTINCT {quote_identifier(column)} FROM {quote_identifier(table)} "
                    f"WHERE typeof({quote_identifier(column)}) = 'text' "
                    f"AND length({quote_identifier(column)}) BETWEEN 2 AND {max_length} "
                    f"LIMIT {max_values_per_column}"
                )
                try:
                    values = connection.execute(query).fetchall()
                except sqlite3.Error:
                    continue
                for (value,) in values:
                    index.setdefault(value.strip().lower(), set()).add((table, column, value))
    finally:
        connection.close()
    return index


def candidate_phrases(text: str, max_words: int = 4) -> list[str]:
    quoted = re.findall(r"['\"`]([^'\"`]{2,60})['\"`]", text)
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9.&'\-]*", text)
    phrases = [phrase.strip() for phrase in quoted]
    for size in range(max_words, 0, -1):
        for start in range(len(words) - size + 1):
            chunk = words[start:start + size]
            if size == 1 and (len(chunk[0]) < 3 or chunk[0].lower() in STOPWORDS):
                continue
            if all(word.lower() in STOPWORDS for word in chunk):
                continue
            phrases.append(" ".join(chunk))
    seen = set()
    unique = []
    for phrase in phrases:
        key = phrase.lower().strip(" .'")
        if key and key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def matched_values(db_path: Path, question: str, evidence: str = "", limit: int = 12) -> str:
    index = value_index(db_path)
    matches: list[tuple[str, str, str]] = []
    covered: list[str] = []
    for phrase in candidate_phrases(f"{question} {evidence}"):
        if phrase not in index or any(phrase in longer for longer in covered):
            continue
        covered.append(phrase)
        for table, column, value in sorted(index[phrase])[:3]:
            matches.append((table, column, value))
        if len(matches) >= limit:
            break
    if not matches:
        return ""
    lines = [f"  {quote_identifier(table)}.{quote_identifier(column)} = '{value}'" for table, column, value in matches[:limit]]
    return "Values from the question found in the database:\n" + "\n".join(lines)
