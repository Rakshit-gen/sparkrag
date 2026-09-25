import pytest

from sparkrag.loader import find_documents, load_documents


def test_find_documents_only_returns_supported_extensions(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "notes.md").write_text("# hello")
    (tmp_path / "image.png").write_bytes(b"not text")

    found = {p.split("/")[-1] for p in find_documents(str(tmp_path))}

    assert found == {"notes.txt", "notes.md"}


def test_load_documents_raises_on_empty_dir(spark, tmp_path):
    with pytest.raises(ValueError):
        load_documents(spark, str(tmp_path))


def test_load_documents_reads_file_contents(spark, tmp_path):
    (tmp_path / "a.txt").write_text("some content here")

    df = load_documents(spark, str(tmp_path))
    row = df.first()

    assert "some content here" in row.text
