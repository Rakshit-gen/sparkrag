from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, FloatType

from sparkrag.config import EMBEDDING_MODEL

_model = None


def _get_model():
    """Load the embedding model once per worker process, not once per row.

    sentence-transformers is expensive to load, so a module-level cache
    keyed by process lifetime means every row a worker handles reuses the
    same loaded model instead of reloading it from disk each time.

    Plain UDF on purpose, not pandas_udf: pandas_udf goes through Arrow,
    and Arrow's direct-memory access breaks under JDK 17+ without JVM
    flags that turned out to be more trouble than they were worth for a
    local demo pipeline.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        # cpu on purpose: Spark worker subprocesses don't have a usable
        # Metal/CUDA context, torch's GPU path just hangs or crashes there.
        _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _model


def _embed_one(text: str) -> list[float]:
    model = _get_model()
    return model.encode(text, show_progress_bar=False).tolist()


embed_udf = udf(_embed_one, ArrayType(FloatType()))


def embed_chunks(df: DataFrame) -> DataFrame:
    return df.withColumn("embedding", embed_udf(col("chunk")))
