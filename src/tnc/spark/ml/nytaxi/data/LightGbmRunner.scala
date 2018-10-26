package tnc.spark.ml.nytaxi.data

import com.microsoft.ml.spark.LightGBMRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.Param
import org.apache.spark.sql.DataFrame


class LightGbmRunner(data: DataFrame, featureCols: Array[String], targetCol:String, nPartitions:Int) {
  val vectorAssembler = new VectorAssembler()
    .setInputCols(featureCols)
    .setOutputCol("features")

  val evaluator = new RegressionEvaluator()
    .setLabelCol("fare_amount")
    .setPredictionCol("prediction")
    .setMetricName("rmse")

  val Array(split80, split20) = data.randomSplit(Array(0.8, 0.2), 125)

  val testSet = split20.cache()
  val trainingSet = split80.cache()

  val trainInput = vectorAssembler.transform(trainingSet).select("features", "fare_amount")
    .repartition(nPartitions)
    .cache()
  val testInput = vectorAssembler.transform(testSet).select("features", "fare_amount")
    .repartition(nPartitions)
    .cache()

  val regressorModel = new LightGBMRegressor()
    .setLearningRate(0.075)
    .setMaxDepth(10)
    .setNumIterations(400)
    .setLabelCol(targetCol)
    .setObjective("regression")
    .setFeaturesCol("features")
    .setNumLeaves(160)
    //.setBaggingFraction(1.0)
    //.setBaggingFreq(5)
    //.setMinSumHessianInLeaf(0.0006)
    //.setDefaultListenPort(24001)
    //.setMaxBin(128)
    .setTimeout(100)
    //.setParallelism("voting_parallel")
    .fit(trainInput)

  val predictions = regressorModel.transform(testInput)
  val metric = evaluator.evaluate(predictions)
}