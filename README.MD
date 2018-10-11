# how to run the sample code:

First obtain the data from Kagel website: 
https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data

Put the tran.csv and test.csv in the same folder. e.g. /home/user/data/nyTaxyData/

Complie and prepare the jar to be handed over to spark. 

```code
sbt compile
sbt assembly
```

Note the location where the assemblies are copied. e.g /home/user/scala-2.11/NyTaxiFare-assembly-1.0.jar

To run in local mode , use the following command. Note that you have to pass some parameters as runtime args.

Parmeters to pass:
1) Data location : e.g. /home/user/data/nyTaxyData/
2) Sample of the data to use : e.g 0.5
3) number of partitions before the Lightgbm process starts : e.g. 16

I run into an issue with lightgbm when I have the sameple size set to 0.5 or larger. 

```code
$SPARK_HOME/bin/spark-submit \
  --class tnc.spark.ml.nytaxi.data.processMain \
  --master local[6] \
  --driver-memory 42g \
  /home/user/scala-2.11/NyTaxiFare-assembly-1.0.jar \
  /home/user/data/nyTaxyData/\
  0.5\
  16
```
