def test_write_chunks_upserts_precomputed_embeddings(tmp_path, monkeypatch):
    monkeypatch.setattr("sparkrag.store.CHROMA_DIR", str(tmp_path))
    from sparkrag.store import get_vectorstore, write_chunks

    rows = [
        {"source": "a.txt", "chunk": "hello world", "chunk_id": 1, "embedding": [0.1, 0.2, 0.3]},
        {"source": "a.txt", "chunk": "second chunk", "chunk_id": 2, "embedding": [0.4, 0.5, 0.6]},
    ]

    count = write_chunks(rows)
    assert count == 2

    store = get_vectorstore()
    result = store._collection.get(ids=["a.txt::1", "a.txt::2"])
    assert set(result["documents"]) == {"hello world", "second chunk"}
