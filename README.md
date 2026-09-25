# Self-Correcting Text-to-SQL Agent

I'm building an agent that turns plain-English questions into SQL, runs the query, and fixes its own mistakes. I measure it on the [BIRD mini-dev](https://github.com/bird-bench/mini_dev) benchmark (500 questions over 11 real SQLite databases).

## Results

Fixed 200-question subset, stratified by difficulty (seed 42), `gemini-3.5-flash-lite`, free tier.

| Variant | Exec. accuracy | Exec. errors | Input tokens/query | Latency |
|---|---|---|---|---|
| Single-shot baseline | 58.5% | 1.5% | 1,077 | 6.9s |
| + 3 sample values per column | 60.5% | 0.5% | 2,915 | 6.4s |
| + column descriptions | | | | |
| + values matched from the question | | | | |
| + descriptions and matched values | | | | |
| + self-correction | | | | |

**Execution accuracy** means my query returns the same set of rows as the reference query.

Sample values gave +2 points, but it is not significant: 14 questions fixed, 10 broken, McNemar p = 0.54. It also used 2.7x the input tokens. Random sample values mostly add noise, so I tried two more targeted kinds of context next.

## Setup

```bash
git clone https://github.com/Tejas-1701/text-to-sql-agent.git
cd text-to-sql-agent
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env               # Windows: copy .env.example .env
```

Open `.env` and paste your Gemini API key after `GEMINI_API_KEY=`. You can get a free key at [Google AI Studio](https://aistudio.google.com/apikey).

### Get the data

```bash
./scripts/download_data.sh
```

On Windows, download [minidev.zip](https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip) and unzip it into a `data/` folder. The code finds `mini_dev_sqlite.json` and `dev_databases/` anywhere inside `data/`.

## Usage

```bash
sql-agent run --limit 5                       # quick smoke test
sql-agent run --variant baseline              # 200-question subset
sql-agent run --variant descriptions
sql-agent run --provider ollama --model qwen2.5-coder:7b --rpm 0
sql-agent summary                             # prints the results table
sql-agent compare results\A.jsonl results\B.jsonl   # fixed/broken counts and p-value
sql-agent preview 1471 --variant descriptions_value_match   # show a prompt, no API call
```

Each run writes one line per question to `results/`. If a run stops because of rate limits, running the same command again picks up where it left off.

| Option | Default | Meaning |
|---|---|---|
| `--model` | `gemini-3.5-flash-lite` | Any Gemini or Ollama model name |
| `--subset` | `200` | Size of the fixed subset (`0` = all 500) |
| `--rpm` | `10` | Requests per minute, set to stay under the free-tier limit |
| `--input-price`, `--output-price` | `0` | USD per million tokens, used for the cost column |

## How it works

1. **Schema:** read the `CREATE TABLE` statements.
2. **Extra context**, depending on the variant:
   - `schema_values`: 3 example values for every column.
   - `descriptions`: what each column means and what its codes stand for, from BIRD's `database_description` files.
   - `value_match`: words from the question and hint that appear as real values in the database, with their table and column.
   - `descriptions_value_match`: both of the above.
3. **Generate:** ask the model for one SQLite query.
4. **Execute:** run it read-only with a 30-second timeout.
5. **Compare:** check the result rows against the reference query.

I compare variants question by question with `sql-agent compare` and McNemar's exact test, because a small change in overall accuracy can hide many questions that flipped in both directions.

Coming next: a retry loop that sends errors, empty results and suspicious results back to the model, and majority voting over several candidate queries.

## Project layout

```
src/sql_agent/
  dataset.py    load BIRD questions, build the fixed subset
  schema.py     describe tables with sample values
  context.py    column descriptions and value matching
  executor.py   read-only SQL execution with timeout
  llm.py        Gemini and Ollama clients with rate limiting and retries
  agent.py      agent variants
  evaluate.py   evaluation loop, metrics, results table
  cli.py        command-line entry point
tests/          tests on a small stand-in database, no API calls
```

## Tests

```bash
pytest
```
