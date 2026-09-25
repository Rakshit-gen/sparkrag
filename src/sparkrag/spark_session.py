from pyspark.sql import SparkSession


def get_spark(app_name: str = "sparkrag") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master("local[2]")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
