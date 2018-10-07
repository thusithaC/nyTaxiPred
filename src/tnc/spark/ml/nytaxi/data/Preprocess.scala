package tnc.spark.ml.nytaxi.data

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassifier, XGBoostRegressor}
import org.apache.spark.ml.tuning.CrossValidator


class Preprocess(implicit spark:SparkSession) {

  import spark.implicits._

  val basePath = "/home/thusitha/work/bigdata/datasets/nytaxi/"
  //val basePath = "hdfs://vmhost:54310/nytaxi/"
  val trainDataPath = basePath + "train.csv"
  val testDataPath = basePath + "test.csv"

  val dataSchemaTrain = StructType(Array(
    StructField("key", StringType, false),
    StructField("fare_amount", DoubleType, false),
    StructField("pickup_datetime", StringType, false),
    StructField("pickup_longitude", DoubleType, false),
    StructField("pickup_latitude", DoubleType, false),
    StructField("dropoff_longitude", DoubleType, false),
    StructField("dropoff_latitude", DoubleType, false),
    StructField("passenger_count", IntegerType, false)
    ))
  val dataSchemaTest = StructType(Array(
    StructField("key", StringType, false),
    StructField("pickup_datetime", StringType, false),
    StructField("pickup_longitude", DoubleType, false),
    StructField("pickup_latitude", DoubleType, false),
    StructField("dropoff_longitude", DoubleType, false),
    StructField("dropoff_latitude", DoubleType, false),
    StructField("passenger_count", IntegerType, false)
  ))

  val rawTrainData = spark.read.format("com.databricks.spark.csv")
    .schema(dataSchemaTrain)
    .option("header", "true")
    .load(trainDataPath)
    .sample(0.4)
    .na.drop()

  val rawTestData = spark.read.format("com.databricks.spark.csv")
    .schema(dataSchemaTest)
    .option("header", "true")
    .load(testDataPath)
    .withColumn("pickup_datetime", unix_timestamp($"pickup_datetime", "yyyy-MM-dd HH:mm:ss").cast(TimestampType))

  val rawTrainDataFiltered = rawTrainData
    .withColumn("pickup_datetime", unix_timestamp($"pickup_datetime", "yyyy-MM-dd HH:mm:ss").cast(TimestampType))
    .filter($"pickup_longitude" >= -75 and $"pickup_longitude" <= -73)
    .filter($"dropoff_longitude" >= -75 and $"dropoff_longitude" <= -73)
    .filter($"pickup_latitude" >= 39 and $"pickup_latitude" <= 42)
    .filter($"dropoff_latitude" >= 39 and $"dropoff_latitude" <= 42)
    .filter($"fare_amount" >= 0 and $"fare_amount" <= 275)
    .filter($"dropoff_longitude" =!= $"pickup_longitude" and $"dropoff_latitude" =!= $"pickup_latitude")

  val trainDataEngineered = rawTrainDataFiltered
    .withColumn("month", month($"pickup_datetime"))
    .withColumn("dayofmonth", dayofmonth($"pickup_datetime"))
    .withColumn("dayofweek", dayofweek($"pickup_datetime"))
    .withColumn("dayofyear", dayofyear($"pickup_datetime"))
    .withColumn("hour", hour($"pickup_datetime"))
    .withColumn("quarter", quarter($"pickup_datetime"))
    .withColumn("year", year($"pickup_datetime")-2000)
    .withColumn("is_night", ($"hour" >= 17 and $"hour" < 21).cast(IntegerType))
    .withColumn("is_late_night", ($"hour" >= 21 and $"hour" < 5).cast(IntegerType))
    .withColumn("is_weekday", ($"dayofweek" > 1 and $"dayofweek" < 7).cast(IntegerType))// end of date manipulations
    .withColumn("latdiff", ($"dropoff_latitude" - $"pickup_latitude"))
    .withColumn("londiff", ($"dropoff_longitude" - $"pickup_longitude"))
    .withColumn("euclidean", sqrt($"latdiff"*$"latdiff" + $"londiff"*$"londiff"))
    .withColumn("manhattan", abs($"latdiff") + abs($"londiff"))
    .drop($"key")
    .na.drop()

  val testDataEngineered = rawTestData
    .withColumn("month", month($"pickup_datetime"))
    .withColumn("dayofmonth", dayofmonth($"pickup_datetime"))
    .withColumn("dayofweek", dayofweek($"pickup_datetime"))
    .withColumn("dayofyear", dayofyear($"pickup_datetime"))
    .withColumn("hour", hour($"pickup_datetime"))
    .withColumn("quarter", quarter($"pickup_datetime"))
    .withColumn("year", year($"pickup_datetime")-2000)
    .withColumn("is_night", ($"hour" >= 17 and $"hour" < 21).cast(IntegerType))
    .withColumn("is_late_night", ($"hour" >= 21 and $"hour" < 5).cast(IntegerType))
    .withColumn("is_weekday", ($"dayofweek" > 1 and $"dayofweek" < 7).cast(IntegerType))// end of date manipulations
    .withColumn("latdiff", ($"dropoff_latitude" - $"pickup_latitude"))
    .withColumn("londiff", ($"dropoff_longitude" - $"pickup_longitude"))
    .withColumn("euclidean", sqrt($"latdiff"*$"latdiff" + $"londiff"*$"londiff"))
    .withColumn("manhattan", abs($"latdiff") + abs($"londiff"))


  val trainColumns = trainDataEngineered.columns
    .filter(_ != "fare_amount")
    .filter(_ != "pickup_datetime")
  val targetCol = "fare_amount"

  val vectorAssembler = new VectorAssembler().
    setInputCols(trainColumns).
    setOutputCol("features")

  /*
  val xgbTrainInput = vectorAssembler.transform(trainDataEngineered).select("features", "fare_amount")


  val xgbParam = Map("eta" -> 0.1f,
    "max_depth" -> 10,
    "objective" -> "reg:linear",
    "num_round" -> 1000,
    "num_workers" -> 8)

  val xgbRegressor = new XGBoostRegressor(xgbParam)
    .setFeaturesCol("features")
    .setLabelCol("fare_amount")

  val xgbREgressionModel = xgbRegressor.fit(xgbTrainInput)

*/

}

object processMain extends App {
  implicit val spark:SparkSession = SparkSession
    .builder()
    .appName("NyTaxiFair")
    .master("local[7]")
    .getOrCreate()

  /*    //.master("local[7]")
    //.config("spark.driver.memory", "24G")
  * */

  val preprocess = new Preprocess
  preprocess.trainDataEngineered.show(100)

  val gbRunner = new LightGbmRunner(preprocess.trainDataEngineered, preprocess.trainColumns, preprocess.targetCol)
  //val model = gbRunner.cvModel
  println("testing score " + gbRunner.metric)
  //println(gbRunner.bestParams)

}

/*
* */
