# RAG Evaluation & Benchmarking Pipeline

A config-driven harness that systematically measures retrieval precision, answer
faithfulness, and hallucination rate across chunk size, chunk overlap, top-k and
embedding configurations using [RAGAS](https://docs.ragas.io) on a LangChain +
ChromaDB RAG stack. Works **100% free and locally with [Ollama](https://ollama.com)**
(out of the box) or with the paid OpenAI API (set `provider: openai`).

## What it does

1. Loads a knowledge base from `data/sample_docs/` (replace with your own `.txt`/`.md`/`.pdf`).
2. For each experiment in `config.yaml`, builds a fresh ChromaDB index with the configured
   chunk size / overlap / embedding model and answers every question in `data/test_set.json`.
3. Scores each run with RAGAS LLM-as-judge metrics:
   - `context_precision` - are the retrieved chunks actually relevant? (LLM-graded)
   - `context_recall` - did retrieval surface the ground-truth information?
   - `faithfulness` - is every claim in the answer supported by the retrieved context?
   - `answer_relevancy` - how well does the answer address the question?
4. Records every run (config + scores) into `results/benchmark.db` and exports
   `results/results.csv` + `results/per_question.csv` so changes can be compared
   across iterations and regressions caught before shipping.

## Setup

### Option A - Free, local with Ollama (default, no API key)

1. Install [Ollama](https://ollama.com/download) and start it (it runs in your
   system tray once installed).
2. Pull the two models used by `config.yaml`:

   ```bash
   ollama pull llama3.2:3b        # answer generator + RAGAS judge
   ollama pull nomic-embed-text   # embeddings
   ```

3. Install the Python stack:

   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   No `.env` file is needed - the Ollama provider sends `base_url`
   `http://localhost:11434/v1` and a dummy `api_key`. `config.yaml` already
   defaults to `provider: ollama`.

### Option B - Paid OpenAI API

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # then paste your OPENAI_API_KEY
```

Then in `config.yaml` switch to `provider: openai` (and use an OpenAI model such
as `gpt-4o-mini` / `text-embedding-3-small`). Requires billing credits on the
OpenAI account.

> `requirements.txt` pins the versions verified in this repo (works on Python 3.14).
> `langchain-community==0.3.31` is deliberately pinned: newer builds dropped a module
> that RAGAS still imports. The exact model names in `config.yaml` (e.g.
> `llama3.2:3b`) must match what `ollama list` shows - Ollama's OpenAI-compatible
> endpoint is strict about the tag.

## Web app (upload docs + chat)

A beginner-friendly web UI ships in this repo - upload your own documents, click
"Build index", then chat with them (local free RAG via Ollama):

```bash
.venv\Scripts\python -m streamlit run app.py
```

Then open http://localhost:8501. To share it with the public (free for visitors, on
a small server), follow `DEPLOY.md`.

## Scores dashboard (browser)

A Streamlit page that reads `results/benchmark.db` and shows your RAGAS scores as
tables + bar charts:

```bash
.venv\Scripts\python -m streamlit run dashboard.py
```

Then open http://localhost:8501 (or `--server.port 8502` to run it next to the chat app).

## Usage

### 1. Smoke test (no API key, free)

Verify the whole chunking -> indexing -> retrieval harness offline with local
deterministic embeddings:

```bash
.venv\Scripts\python run_benchmark.py --smoke
```

Reports `reference_hit_rate` (share of questions whose best retrieved chunk shares
>= 15% tokens with the ground-truth answer) and average token-overlap Jaccard.

### 2. Full RAGAS benchmark

```bash
.venv\Scripts\python run_benchmark.py
```

Runs every experiment in `config.yaml`, prints a comparison table, and saves results.

Useful flags:

```bash
--experiments baseline-600-80 small-300-50   # run a subset only
--questions 4                                # quick run on the first N questions
--history                                    # review all past runs from the DB
```

## Adding your own data

- Drop documents into `data/sample_docs/` (subfolders supported).
- Add/modify questions in `data/test_set.json`. Fields: `question`, `reference`
  (ground-truth answer, used by `context_recall` / `context_precision`). The last
  question in the sample set demonstrably has no answer in the corpus - keep one
  such "unanswerable" question to surface hallucination behavior.
- Define experiments in `config.yaml` under `experiments` (each overrides
  `chunk_size`, `chunk_overlap`, `top_k`).

## Project layout

```
config.yaml            benchmark settings + experiment grid
run_benchmark.py       CLI entry point
rag/                   corpus loading, chunking, embeddings, Chroma RAG pipeline
evaluation/            test-set loading + RAGAS evaluation wrapper
benchmark/             experiment runner, SQLite/CSV results store, report tables
data/sample_docs/      sample knowledge base
data/test_set.json     ground-truth test questions
results/               benchmark.db, results.csv, per_question.csv (gitignored)
```

## Interpreting results

Scores are 0-1, higher is better. Typical trade-offs the grid exposes:

| Signal | Interpretation |
| --- | --- |
| `context_precision` drops | too much irrelevant text per chunk (raise overlap? lower top-k?) |
| `context_recall` drops | ground truth missing from top-k (lower `chunk_size` / raise `top_k`) |
| `faithfulness` drops | answer hallucinates or quotes noise outside context |
| `answer_relevancy` drops | retrieval surfaced wrong content entirely |

Example: `context_recall` up + `context_precision` down when `top_k` increases is the
classic precision/recall tension - pick the config best for your application, not the
single best average.