from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, explode, monotonically_increasing_id, udf
from pyspark.sql.types import ArrayType, StringType

from sparkrag.config import CHUNK_OVERLAP, CHUNK_SIZE


def _split(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_text(text)


split_udf = udf(_split, ArrayType(StringType()))


def chunk_documents(df: DataFrame) -> DataFrame:
    """Explode each document row into one row per chunk.

    Runs as a Spark UDF so the splitting happens in parallel across
    whatever partitions the loader produced.
    """
    chunked = df.withColumn("chunks", split_udf(col("text"))).select(
        "source", explode(col("chunks")).alias("chunk")
    )
    return chunked.withColumn("chunk_id", monotonically_increasing_id())
