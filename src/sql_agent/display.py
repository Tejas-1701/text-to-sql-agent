def format_table(columns: list[str], rows: list[tuple], limit: int = 20, width: int = 40) -> str:
    if not rows:
        return "(no rows)"
    shown = [[str(value) if len(str(value)) <= width else str(value)[: width - 3] + "..." for value in row] for row in rows[:limit]]
    headers = list(columns) if columns else [f"column_{index + 1}" for index in range(len(shown[0]))]
    widths = [max(len(headers[index]), *(len(row[index]) for row in shown)) for index in range(len(headers))]
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * size for size in widths),
    ]
    lines.extend("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) for row in shown)
    if len(rows) > limit:
        lines.append(f"... {len(rows) - limit} more rows")
    return "\n".join(lines)
