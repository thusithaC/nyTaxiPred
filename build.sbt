name := "NyTaxiFare"

version := "1.0"

scalaVersion := "2.11.6"

fork := true

val sparkVersion = "2.3.2"

resolvers ++= Seq(
  "apache-snapshots" at "http://repository.apache.org/snapshots/",
  "Xgb GitHub Repo" at "https://raw.githubusercontent.com/CodingCat/xgboost/maven-repo/",
  "Maven scala repo" at "https://dl.bintray.com/spark-packages/maven",
  "Azure" at "https://mmlspark.azureedge.net/maven"
)

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)
assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

scalaSource in Compile := baseDirectory.value / "src"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.3.2",
  "org.apache.spark" %% "spark-sql" % "2.3.2",
  "com.microsoft.ml.spark" % "mmlspark_2.11" % "0.14.dev13+1.g58a2027c",

  //"Azure" % "mmlspark" % "0.14",

//"com.johnsnowlabs.nlp" %% "spark-nlp" % "1.5.4",
  "org.apache.spark" %% "spark-mllib" % "2.3.2"
  //"org.apache.spark" %% "spark-mllib" % sparkVersion
  //"org.apache.spark" %% "spark-streaming" % sparkVersion
  //"org.apache.spark" %% "spark-hive" % sparkVersion
)
