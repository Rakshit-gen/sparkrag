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
