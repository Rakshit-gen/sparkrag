# sparkrag

Most RAG demos load a folder of twenty text files into memory and call it done.
That falls apart the moment you point it at something real: a support team's
ticket archive, a company wiki export, a few hundred thousand PDFs. Reading,
cleaning, chunking, and embedding that much text on a single thread takes
hours and usually falls over on memory before it finishes.

sparkrag uses PySpark to do that work in parallel across partitions, then
serves the result through a normal LangChain retrieval chain backed by Groq.
It runs in local mode (`local[*]`) so it works fine on a laptop, but the
ingestion side scales the same way if you point it at a cluster later.

## What it does

1. Reads a directory of `.txt`, `.md`, and `.pdf` files into a Spark
   DataFrame, one row per file.
2. Splits each file into overlapping chunks.
3. Embeds each chunk with a local sentence-transformers model, loaded once
   per Spark partition instead of once per row.
4. Writes the chunks and their embeddings into a local Chroma store.
5. Answers questions against that store using Groq as the LLM.

## Setup

Use Python 3.11. PySpark's cloudpickle does not handle 3.13+ correctly yet
(you'll get a pickling RecursionError on `createDataFrame`), so newer
interpreters won't work here.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
# put your GROQ_API_KEY in .env
```

## Usage

```bash
# ingest a folder of documents
python -m sparkrag.cli ingest --input ./sample_docs

# ask a question against what you ingested
python -m sparkrag.cli query "what does the onboarding doc say about VPN access?"
```

Sample documents are included in `sample_docs/` so the pipeline is runnable
without bringing your own data first.

## Why Spark and not just a for loop

A for loop over files is fine until the folder stops being small. Spark
gives us parallel reads, parallel chunking, and parallel embedding for free,
and the same code path handles ten files or ten million without a rewrite.
The embedding step in particular is the expensive part, so `embedder.py`
loads the model once per partition and reuses it for every row in that
partition instead of reloading it per document.

## Troubleshooting

**`PicklingError` / `RecursionError` on `createDataFrame`**: you're on
Python 3.13+. Use 3.11.

**`sun.misc.Unsafe ... not available`**: this only comes up if you switch
the embedder back to a `pandas_udf`. Arrow's direct memory access breaks
under JDK 17+ without JVM `--add-opens` flags that aren't worth the
trouble for a local pipeline, which is why `embedder.py` uses a plain UDF
instead.

**`ModuleNotFoundError: No module named 'sparkrag'` from a Spark worker**:
you skipped `pip install -e .`. Spark workers import your code by name,
they don't inherit the driver's `sys.path`.

## Production considerations

This is a working pipeline, not a toy demo, but "production" covers a lot of
ground. Here's what's actually handled and what isn't.

**Handled:**

- **Docker**: a `Dockerfile` builds a python:3.11-slim image with
  openjdk-17-jre-headless installed for pyspark, `JAVA_HOME` set correctly,
  and the package installed editable so the entrypoint runs `sparkrag.cli`.
- **CI**: `.github/workflows/ci.yml` runs the test suite on every push and
  pull request, with Python 3.11 and a Temurin JDK set up so the Spark-based
  tests actually run instead of failing on a missing `JAVA_HOME`.
- **Structured logging**: `cli.py` logs status (chunk counts, errors) through
  Python's `logging` module at INFO/ERROR instead of bare `print`, so output
  can be filtered or shipped somewhere. The actual query answer stays on
  plain stdout so it's still pipeable.
- **Retries**: the Groq call in `chain.py` retries up to 3 times with
  exponential backoff on failure. A missing API key still fails immediately,
  it's checked before the retried call runs.
- **Input validation**: a bad or missing `--input` directory produces a
  one-line error, not a Spark stack trace.

**Not handled, and worth knowing before you'd actually ship this:**

- **Spark still runs in local mode.** `local[2]` on one machine, not a real
  cluster. The loader/chunker/embedder code is structured so it *could* run
  on YARN or Kubernetes with a config change, but that's untested here and
  would need its own tuning (partition counts, executor memory, etc).
- **No auth, anywhere.** The CLI has no concept of users. If you wrapped
  this in an API, you'd need to add auth, rate limiting, and probably
  per-user document isolation before letting anyone else near it.
- **Chroma is a local, single-writer store.** Fine for one process on one
  disk. It's not going to hold up as a shared, concurrent-write vector
  store; a real deployment would want a hosted vector DB (Chroma's own
  server mode, pgvector, Pinecone, etc).
- **No secrets management.** `GROQ_API_KEY` comes from a `.env` file. That's
  fine for local dev, not for a deployed container, which should pull it
  from a real secrets manager or the orchestrator's secret store.
- **No observability beyond logs.** There's no metrics, tracing, or
  alerting on ingest failures, retry exhaustion, or latency. You'd want at
  least basic metrics before running this unattended.
- **The Docker build hasn't been run in this environment** (no Docker
  available), so it's written correctly based on the actual dependencies in
  `requirements.txt` but hasn't been build-tested end to end.

## Layout

```
src/sparkrag/
  config.py     env vars and constants
  spark_session.py   Spark session builder
  loader.py     reads txt/md/pdf into a Spark DataFrame
  chunker.py    splits text into chunks
  embedder.py   partition-local embedding
  store.py      Chroma wrapper
  ingest.py     wires loader -> chunker -> embedder -> store together
  chain.py      Groq-backed retrieval chain
  cli.py        ingest / query commands
```
