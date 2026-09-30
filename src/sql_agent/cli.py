import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .agent import VARIANTS, build_agent
from .chart import collect_points, draw_chart
from .dataset import database_path, find_databases_dir, fixed_subset, load_examples
from .display import format_table
from .errors import export_errors
from .evaluate import compare_runs, markdown_table, run_evaluation, summarize
from .llm import build_model


def run_command(arguments) -> None:
    data_root = Path(arguments.data)
    examples = load_examples(data_root)
    examples = fixed_subset(examples, arguments.subset, arguments.seed) if arguments.subset else examples
    if arguments.limit:
        examples = examples[: arguments.limit]
    model = build_model(arguments.provider, arguments.model, arguments.rpm)
    agent = build_agent(arguments.variant, model)
    safe_model = arguments.model.replace("/", "-").replace(":", "-")
    output_path = Path(arguments.out) / f"{arguments.variant}__{safe_model}__n{len(examples)}.jsonl"
    run_evaluation(agent, examples, find_databases_dir(data_root), output_path)
    print(json.dumps(summarize(output_path, arguments.input_price, arguments.output_price), indent=2))


def summary_command(arguments) -> None:
    paths = sorted(Path(arguments.results).glob("*.jsonl"))
    summaries = [summarize(path, arguments.input_price, arguments.output_price) for path in paths]
    print(markdown_table(summaries))


def compare_command(arguments) -> None:
    result = compare_runs(Path(arguments.first), Path(arguments.second))
    print(f"{result['first']}  vs  {result['second']}  ({result['shared_questions']} shared questions)")
    print(f"accuracy: {result['first_accuracy']:.1%} -> {result['second_accuracy']:.1%}")
    print(f"fixed: {result['fixed_by_second']}  broken: {result['broken_by_second']}  "
          f"both correct: {result['both_correct']}  both wrong: {result['both_wrong']}")
    print(f"McNemar exact p = {result['mcnemar_p']:.3f}")


def errors_command(arguments) -> None:
    data_root = Path(arguments.data)
    output_path = Path(arguments.out)
    flags = export_errors(
        [Path(path) for path in arguments.runs],
        data_root,
        find_databases_dir(data_root),
        output_path,
        arguments.sample,
    )
    total = sum(flags.values())
    print(f"{total} questions were wrong in every run -> {output_path}")
    for flag, count in flags.most_common():
        print(f"  {count:>3}  {flag}")


def ask_command(arguments) -> None:
    databases_dir = find_databases_dir(Path(arguments.data))
    db_path = database_path(databases_dir, arguments.db)
    if not db_path.exists():
        available = sorted(folder.name for folder in databases_dir.iterdir() if folder.is_dir())
        raise SystemExit(f"Unknown database '{arguments.db}'. Choose one of: {', '.join(available)}")
    model = build_model(arguments.provider, arguments.model, arguments.rpm)
    agent = build_agent(arguments.variant, model)
    result = agent.answer(db_path, arguments.question, arguments.hint)
    print(f"SQL:\n{result.sql}\n")
    if result.execution.error:
        print(f"The query failed: {result.execution.error}")
    else:
        print(format_table(result.execution.columns, result.execution.rows))
    print(f"\n{result.llm_calls} model call(s), {result.llm_seconds:.1f}s")


def chart_command(arguments) -> None:
    points = collect_points(Path(arguments.results), arguments.subset)
    output_path = draw_chart(points, Path(arguments.out))
    print(f"Charted {len(points)} runs -> {output_path}")


def preview_command(arguments) -> None:
    data_root = Path(arguments.data)
    matches = [example for example in load_examples(data_root) if example.question_id == arguments.question_id]
    if not matches:
        raise SystemExit(f"No question with id {arguments.question_id}")
    example = matches[0]

    class NoModel:
        name = "none"

    agent = build_agent(arguments.variant, NoModel())
    db_path = database_path(find_databases_dir(data_root), example.db_id)
    prompt = agent.build_prompt(db_path, example.question, example.evidence)
    print(f"System prompt:\n{agent.system_prompt}\n\n{prompt}")
    print(f"\n--- {len(agent.system_prompt) + len(prompt)} characters, about {(len(agent.system_prompt) + len(prompt)) // 4} tokens ---")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="sql-agent")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="Run an agent variant on BIRD mini-dev")
    run_parser.add_argument("--data", default="data")
    run_parser.add_argument("--provider", choices=["gemini", "ollama"], default="gemini")
    run_parser.add_argument("--model", default="gemini-3.5-flash-lite")
    run_parser.add_argument("--variant", choices=list(VARIANTS), default="baseline")
    run_parser.add_argument("--subset", type=int, default=200)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--limit", type=int, default=0)
    run_parser.add_argument("--rpm", type=float, default=10.0)
    run_parser.add_argument("--out", default="results")
    run_parser.add_argument("--input-price", type=float, default=0.0)
    run_parser.add_argument("--output-price", type=float, default=0.0)
    run_parser.set_defaults(handler=run_command)

    summary_parser = commands.add_parser("summary", help="Print a results table for all runs")
    summary_parser.add_argument("--results", default="results")
    summary_parser.add_argument("--input-price", type=float, default=0.0)
    summary_parser.add_argument("--output-price", type=float, default=0.0)
    summary_parser.set_defaults(handler=summary_command)

    compare_parser = commands.add_parser("compare", help="Compare two runs question by question")
    compare_parser.add_argument("first")
    compare_parser.add_argument("second")
    compare_parser.set_defaults(handler=compare_command)

    errors_parser = commands.add_parser("errors", help="Export questions that every given run got wrong, for labeling")
    errors_parser.add_argument("runs", nargs="+")
    errors_parser.add_argument("--data", default="data")
    errors_parser.add_argument("--out", default="analysis/errors.csv")
    errors_parser.add_argument("--sample", type=int, default=30)
    errors_parser.set_defaults(handler=errors_command)

    ask_parser = commands.add_parser("ask", help="Ask one question about a database and print the SQL and result")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--db", required=True)
    ask_parser.add_argument("--hint", default="")
    ask_parser.add_argument("--variant", choices=list(VARIANTS), default="self_correct")
    ask_parser.add_argument("--data", default="data")
    ask_parser.add_argument("--provider", choices=["gemini", "ollama"], default="gemini")
    ask_parser.add_argument("--model", default="gemini-3.5-flash-lite")
    ask_parser.add_argument("--rpm", type=float, default=10.0)
    ask_parser.set_defaults(handler=ask_command)

    chart_parser = commands.add_parser("chart", help="Draw accuracy with confidence intervals for all complete runs")
    chart_parser.add_argument("--results", default="results")
    chart_parser.add_argument("--subset", type=int, default=200)
    chart_parser.add_argument("--out", default="results/accuracy.png")
    chart_parser.set_defaults(handler=chart_command)

    preview_parser = commands.add_parser("preview", help="Print the prompt for one question without calling the model")
    preview_parser.add_argument("question_id", type=int)
    preview_parser.add_argument("--variant", choices=list(VARIANTS), default="descriptions_value_match")
    preview_parser.add_argument("--data", default="data")
    preview_parser.set_defaults(handler=preview_command)

    arguments = parser.parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
