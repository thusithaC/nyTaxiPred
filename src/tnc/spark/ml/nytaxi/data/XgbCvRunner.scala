package tnc.spark.ml.nytaxi.data

import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.PipelineModel
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel

class XgbCvRunner(data: DataFrame, featureCols: Array[String], targetCol:String) {

  val vectorAssembler = new VectorAssembler()
    .setInputCols(featureCols)
    .setOutputCol("features")

  val evaluator = new RegressionEvaluator()
    .setLabelCol("fare_amount")
    .setPredictionCol("prediction")
    .setMetricName("rmse")

  val Array(split80, split20) = data.randomSplit(Array(0.8, 0.2), 123)

  val testSet = split20.cache()
  val trainingSet = split80.cache()

  val xgbTrainInput = vectorAssembler.transform(trainingSet).select("features", "fare_amount")
  val xgbTestInput = vectorAssembler.transform(testSet).select("features", "fare_amount")

  val xgbRegressor = new XGBoostRegressor()
    .setFeaturesCol("features")
    .setLabelCol("fare_amount")
    .setNumEarlyStoppingRounds(40)
    .setUseExternalMemory(true)
    .setNumRound(1000)
    .setSilent(0)
    .setNumWorkers(8)


  val paramGrid = new ParamGridBuilder()
    .addGrid(xgbRegressor.eta, Range.Double(0.005,0.010,0.005))
    //.addGrid(xgbRegressor.maxDepth, Range.Int(4, 6, 1))
    //.addGrid(xgbRegressor.minChildWeight, Range.Double(0.05,0.2,0.05))
    //.addGrid(xgbRegressor.gamma, Range.Double(0.00,0.05,0.0125))
    .build()

  val cv = new CrossValidator()
    .setEstimator(xgbRegressor)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(2)
    //.setParallelism(4)


  val cvModel = cv.fit(xgbTrainInput).bestModel

  val predictions = cvModel.transform(xgbTestInput)
  val metric = evaluator.evaluate(predictions)

  val bestParams = cvModel.asInstanceOf[XGBoostRegressionModel].params

}
