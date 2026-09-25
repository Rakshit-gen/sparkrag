from typing import Iterator

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType

from sparkrag.config import EMBEDDING_MODEL

_model = None


def _get_model():
    """Load the embedding model once per worker process, not once per row.

    sentence-transformers is expensive to load (it pulls the model weights
    into memory), so a module-level cache keyed by process lifetime means
    every row in a partition reuses the same loaded model.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


@pandas_udf(ArrayType(FloatType()))
def embed_udf(texts: pd.Series) -> pd.Series:
    model = _get_model()
    vectors = model.encode(texts.tolist(), show_progress_bar=False)
    return pd.Series([v.tolist() for v in vectors])


def embed_chunks(df: DataFrame) -> DataFrame:
    return df.withColumn("embedding", embed_udf(df["chunk"]))
