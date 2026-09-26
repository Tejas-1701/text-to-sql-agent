# Self-Correcting Text-to-SQL Agent

I'm building an agent that turns plain-English questions into SQL, runs the query, and fixes its own mistakes. I measure it on the [BIRD mini-dev](https://github.com/bird-bench/mini_dev) benchmark (500 questions over 11 real SQLite databases).

## Results

Fixed 200-question subset, stratified by difficulty (seed 42), `gemini-3.5-flash-lite`, temperature 0.

| Variant | Exec. accuracy | Exec. errors | Input tokens/query | Latency | Fixed / broken vs baseline | McNemar p |
|---|---|---|---|---|---|---|
| Single-shot baseline | 58.0% | 1.5% | 1,077 | 6.2s | | |
| + 3 sample values per column | 57.0% | 0.0% | 2,915 | 6.2s | 5 / 7 | 0.77 |
| + column descriptions | **61.0%** | 0.5% | 2,441 | 6.1s | 13 / 7 | 0.26 |
| + values matched from the question | 59.0% | 1.0% | 1,114 | 7.4s | 9 / 7 | 0.80 |
| + descriptions and matched values | 58.5% | 1.5% | 2,478 | 8.1s | 9 / 8 | 1.00 |
| + self-correction | | | | | | |

**Execution accuracy** means my query returns the same set of rows as the reference query. Paired comparisons use the 199 unique questions (one BIRD question appears twice).

### What I learned

- **Run-to-run noise is as big as the differences.** I ran the baseline and the sample-values variant twice each with identical settings. The baseline scored 58.5% then 58.0%, and sample values scored 60.5% then 57.0%. Even at temperature 0, the model changes its answer on about 7% of questions between runs.
- **Column descriptions were the only context that pointed the right way.** They fixed 13 questions and broke 7, but that is not significant at this sample size (p = 0.26).
- **Random sample values and value matching did not help.** BIRD's hints already spell out most of the values a question needs, so finding them in the database adds little.
- **Almost every miss is a wrong answer, not a crash.** Only 0 to 3 queries per run failed to execute, so a retry loop that only reacts to errors would barely help.

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
