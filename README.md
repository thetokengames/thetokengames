# The Token Games (TTG)

Code and data accompanying the paper **"The Token Games"** (anonymous submission).

TTG is a competitive arena where two language models take turns inventing and
solving Python code puzzles. In each round one model is the **proposer**: it
writes a function `mystery(x)` that returns a boolean and provides a sample
input that makes it return `True`. The other model is the **solver**: it sees
only the function body and must produce its own input that makes `mystery(x)`
return `True`. The puzzle code is executed in a sandbox to verify both the
proposer's sample and the solver's attempt.

This release contains:

- The arena that runs head-to-head matches between any two models.
- The full set of pairwise match logs used in the paper (`data/`).
- Scripts to compute three Elo ratings (match-level, proposer, solver) from the
  logs and to reproduce the analyses and plots in the paper.
- A puzzle-mining tool that re-runs previously seen puzzles against new models,
  for evaluating models on a fixed set of puzzles without re-running the arena.

---

## Repository layout

```
arena.py          Runs duels between models.
llm_utils.py      Model factory, token tracking, sandboxed puzzle execution.
elo.py            Computes match / proposer / solver Elo from JSONL match logs.
analysis.py       Correlation analysis vs. external benchmarks; failure plots.
mine_puzzles.py   Re-tests previously seen puzzles against arbitrary models.
requirements.txt  Python dependencies.
data/             Match logs (JSONL/log/raw/tags) for every model pair in the paper.
```

### Data format

For each ordered model pair `A_vs_B` in `data/` we release six files:

| Suffix | Contents |
| --- | --- |
| `.jsonl` | One JSON record per round: code, sample, attempt, outcome, token usage, scores. |
| `.log` | Human-readable transcript of the duel. |
| `.raw.txt` | Full prompts and raw model responses (including reasoning traces when exposed by the API). |
| `.tags.jsonl` | Per-round puzzle tags used in the paper's qualitative analysis. |
| `_stdout.txt` / `_stderr.txt` | Captured process output from the run. |

`data/status.json` and `data/summary.txt` summarize completion status and
final scores across all pairs.

### Outcome codes

Every round in `*.jsonl` has one of these `outcome` values:

| Code | Meaning |
| --- | --- |
| `success` | Solver found a valid input -- counts as a draw. |
| `failure` | Solver did not find a valid input -- proposer wins. |
| `puzzle-wrong` | Proposer's own sample input was invalid -- proposer is penalized. |
| `*-ex` | Same as the base outcome, but it was determined by a runtime exception (e.g. timeout) rather than a clean `False`. |

---

## Setup

### Requirements

- Python 3.12+
- Docker, used to sandbox candidate solutions when verifying puzzles.
- API access for whichever providers you intend to use. We use langchain (see dependencies in requirements.txt) for accessing models in a provider-agnostic manner. The model factor in llm_utils.py recognizes:
  - **OpenAI** (`gpt-*`, `o4*`) — `OPENAI_API_KEY`
  - **Anthropic** (`claude-*`) — `ANTHROPIC_API_KEY`
  - **Google** (`gemini-*`) — `GOOGLE_API_KEY`
  - **xAI** (`grok-*`) — `XAI_API_KEY`
  - **DeepSeek** (`deepseek-*` direct) — `DEEPSEEK_API_KEY`
  - **OpenRouter** (e.g. `deepseek-v3.2-thinking`, `kimi-k2.6`) —
    `OPENROUTER_API_KEY`
  - **Ollama** (any model name containing `:`) — local server on
    `http://localhost:11434`

### Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Verify Docker sandbox

The arena runs candidate inputs inside a hardened Python container. Confirm it
works on your system before launching:

```bash
docker run --rm --network none --memory=256m --pids-limit=64 --cpus=1 \
  --read-only -i python:3.12 python --version
```

To run *without* Docker (executing puzzle code directly with the host interpreter -- not recommended for untrusted models), set `SANDBOX=0`.

### API keys

Export keys in your shell, or place them in a `.env` file at the repository
root (loaded automatically via `python-dotenv`):

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
XAI_API_KEY=...
DEEPSEEK_API_KEY=...
OPENROUTER_API_KEY=...
```

---

## Running a duel

```bash
python arena.py --model-a <model-a> --model-b <model-b> --turns 10
```

Each *turn* consists of two rounds: one with model A as proposer, one with
model B as proposer.

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--turns N` | Number of turn pairs (default `5`). |
| `--json-log FILE` | Write the structured JSONL log to `FILE`. |
| `--text-log FILE` | Write the human-readable transcript to `FILE`. |
| `--raw-log FILE` | Write the full prompts and raw responses to `FILE`. |
| `--summary-log FILE` | Append the duel's final score line to `FILE` (default `summary.txt`). |
| `--no-history` | Disable the running history of past puzzles in the proposer prompt. |

If no log paths are given, file names are auto-generated with a timestamp:
`<modelA>_vs_<modelB>_<YYYYMMDD_HHMMSS>.{jsonl,log,raw.txt}`.

A token-usage summary (and an estimated dollar cost, when prices are known in
`MODEL_PRICING`) is printed at the end of every duel.

### Adding a model

`get_llm` in `llm_utils.py` dispatches by prefix. To support a new provider,
add a branch there and (optionally) an entry in `MODEL_PRICING` so cost
estimates show up in the summary. To route a model through OpenRouter, add it
to `OPENROUTER_MODELS`.

---

## Reproducing the paper's analyses

The data released in `data/` is sufficient to reproduce every statistic, table,
ranking, and figure in the paper. The Elo and analysis scripts read JSONL
match logs from `runs/classic/*.jsonl` by default, so first symlink (or copy)
the bundled data into that location:

```bash
mkdir -p runs
ln -s "$PWD/data" runs/classic
```

### Elo ratings

```bash
python elo.py
```

Prints three Elo tables — **match** (one outcome per duel), **proposer**
(per-round, scoring the proposer), and **solver** (per-round, scoring the
solver) — and writes them into `model-scores.json` under the keys
`ttg_match_elo`, `ttg_proposer_elo`, and `ttg_solver_elo`. Elo is fit by
maximum likelihood (gradient descent on the negative log-likelihood of the
observed outcomes).

### Correlations and tables

```bash
python analysis.py correlation
```

Computes Spearman correlations between TTG Elo scores and external benchmarks
listed in `model-scores.json`, prints a LaTeX-ready table, and writes
scatter plots to `plots/`. Use `--exclude REGEX` to drop outlier models
(default excludes one model used as a robustness check in the paper).

### Failure-mode plot

```bash
python analysis.py proposer-failures
```

Writes `plots/proposer_failures.png`, breaking down each model's lost rounds
into "incorrect own solution" vs. "solver succeeded".

### `model-scores.json`

`elo.py` updates this file with computed Elo scores. To rerun the correlation
analysis you also need a `model-scores.json` containing the external benchmark
numbers for each model (e.g. `lmarena`, `livecodebench`, etc.) that you want to
correlate against TTG. The file is a JSON object keyed by model name; each
value is an object mapping benchmark name to score.

---

## Mining puzzles for a fixed-test evaluation

Once you have arena logs you can extract the unique, valid puzzles that were
generated and re-run them against a new set of models. This produces a
fixed-set evaluation (no proposer dynamics) on puzzles previously generated
during self-play.

```bash
python mine_puzzles.py \
  --models gpt-5.2,claude-opus-4-5-20251101 \
  --output mined_results.jsonl \
  data/*.jsonl
```

Each input puzzle is run against each listed model; per-puzzle JSONL records
and an aggregate solve-rate summary are written to `--output`. Re-running the
command with the same `--output` resumes where it left off (puzzle/model
combinations already logged are skipped).

---

## Notes

- All puzzle execution happens in `subprocess` calls to a hardened Python
  container with no network, capped CPU/memory, a 5-second timeout, and a
  read-only root filesystem. Treat untrusted-model output as adversarial; do
  not disable the sandbox unless you fully trust both proposer and solver.
- Reasoning-token usage is reported when the provider exposes it (Anthropic
  thinking blocks, OpenAI/OpenRouter reasoning content, xAI reasoning token
  counts, Google's `thinking_budget`).
- This is a research artifact released for review. We also have a public (non-anonymous) website that we released with the supplemental material in an anonymized version.
