from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, StructType, StructField, DateType, IntegerType, DoubleType
from pyspark.sql.functions import col, expr, year, month

spark = (SparkSession.builder.appName("GoogleStockDataAnalysis").getOrCreate())
source_data_path_csv = "../data/GOOG.csv"

'''
CSV file
'''
print("Before applying the custom schema to the file")
google_df = (
    spark.read.format("csv")
        .option("header", True)
        .load(source_data_path_csv))

google_df.show(10)

print(google_df.printSchema())

'''
Applying the custom schema to the csv file
'''
print("After applying the schema")
schema = StructType([
    StructField("Date", DateType(), True),
    StructField("Open", DoubleType(), True),
    StructField("High", DoubleType(), True),
    StructField("Low", DoubleType(), True),
    StructField("Close", DoubleType(), True),
    StructField("Adj Close", DoubleType(), True),
    StructField("Volume", DoubleType(), True)
])

google_df1 = (
    spark.read.format("csv")
        .option("header", True)
        .schema(schema)
        .load(source_data_path_csv)
)

google_df1.show(10)
print(google_df1.printSchema())

'''
Working with Pyspark Dataframe
'''
# Average closing price per year for google
print("Average closing price per year for google")
google_df1.select(year("Date").alias("year"), "Adj Close").groupby("year").avg("Adj Close").sort("year").show()


print("Computing the average closing price per month for google")

google_df1.select(year("Date").alias("year"),
                month("Date").alias("month"),
                "Adj Close").groupby("year", "month").avg("Adj Close").sort("year", "month").show()

'''
Working with Spark SQL
'''
print("Working with Spark SQL")
google_df1.createOrReplaceTempView("googlesql")
print(spark.catalog.listTables())

googlesql = spark.table('googlesql')
googlesql.show()

print("Difference between low and high")
durationHour = googlesql.withColumn('DifferenceLowHigh', googlesql.High - googlesql.Low)
durationHour.show()
# print(googlesql)



print("Working with Spark SQL")


spark.sql("SELECT googlesql.Date, googlesql.Open, googlesql.Close, abs(googlesql.Close - googlesql.Open) as spydif FROM googlesql WHERE abs(googlesql.Close - googlesql.Open) > 4 ").show()
