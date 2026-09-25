import glob
import os
from typing import Iterator

from pypdf import PdfReader
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StringType, StructField, StructType

SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf")

SCHEMA = StructType(
    [
        StructField("source", StringType(), False),
        StructField("text", StringType(), False),
    ]
)


def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        return _read_pdf(path)
    return _read_text(path)


def find_documents(input_dir: str) -> Iterator[str]:
    for ext in SUPPORTED_EXTENSIONS:
        yield from glob.glob(os.path.join(input_dir, f"**/*{ext}"), recursive=True)


def load_documents(spark: SparkSession, input_dir: str) -> DataFrame:
    """Read every supported file under input_dir into a Spark DataFrame.

    Reading each file happens on the driver (file IO is cheap and PDFs
    don't serialize well), but the resulting rows are spread across
    partitions before any of the expensive chunking or embedding work.
    """
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input path does not exist or is not a directory: {input_dir}")

    rows = []
    for path in find_documents(input_dir):
        text = read_file(path).strip()
        if text:
            rows.append((path, text))

    if not rows:
        raise ValueError(f"No supported documents found under {input_dir}")

    return spark.createDataFrame(rows, schema=SCHEMA).repartition(4)
