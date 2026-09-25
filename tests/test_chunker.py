from sparkrag.chunker import chunk_documents
from sparkrag.loader import SCHEMA


def test_chunk_documents_splits_long_text(spark):
    long_text = "word " * 1000
    df = spark.createDataFrame([("doc1.txt", long_text)], schema=SCHEMA)

    chunks = chunk_documents(df).collect()

    assert len(chunks) > 1
    assert all(row.source == "doc1.txt" for row in chunks)
    assert all(len(row.chunk) <= 900 for row in chunks)


def test_chunk_documents_keeps_short_text_as_one_chunk(spark):
    df = spark.createDataFrame([("doc2.txt", "just a short note")], schema=SCHEMA)

    chunks = chunk_documents(df).collect()

    assert len(chunks) == 1
    assert chunks[0].chunk == "just a short note"
