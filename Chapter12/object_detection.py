# To allow Python to find Spark driver
import findspark
findspark.init('/home/ubuntu/spark-2.4.0-bin-hadoop2.7')

import os
SUBMIT_ARGS = "--packages databricks:spark-deep-learning:1.3.0-spark2.4-s_2.11 pyspark-shell"
os.environ["PYSPARK_SUBMIT_ARGS"] = SUBMIT_ARGS


from pyspark.sql import SparkSession

spark = SparkSession.builder \
      .appName("ImageClassification") \
      .config("spark.executor.memory", "70g") \
      .config("spark.driver.memory", "50g") \
      .config("spark.memory.offHeap.enabled",True) \
      .config("spark.memory.offHeap.size","16g") \
      .getOrCreate()

import pyspark.sql.functions as f
import sparkdl as dl
from pyspark.ml.image import ImageSchema

dfbuses = ImageSchema.readImages('buses/').withColumn('label', f.lit(0))
dfcars = ImageSchema.readImages('cars/').withColumn('label', f.lit(1))

dfbuses.show(5)
dfcars.show(5)

trainDFbuses, testDFbuses = dfbuses.randomSplit([0.60,0.40], seed = 123)
trainDFcars, testDFcars = dfcars.randomSplit([0.60,0.40], seed = 122)


trainDF = trainDFbuses.unionAll(trainDFcars)
testDF = testDFbuses.unionAll(testDFcars)

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
vectorizer = dl.DeepImageFeaturizer(inputCol="image", outputCol="features", 
                    modelName="InceptionV3")
logreg = LogisticRegression(maxIter=30, labelCol="label")
pipeline = Pipeline(stages=[vectorizer, logreg])
pipeline_model = pipeline.fit(trainDF)

predictDF = pipeline_model.transform(testDF)
predictDF.select('prediction', 'label').show(n = testDF.toPandas().shape[0], truncate=False)

predictDF.crosstab('prediction', 'label').show()

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
scoring = predictDF.select("prediction", "label")
accuracy_score = MulticlassClassificationEvaluator(metricName="accuracy")
rate = accuracy_score.evaluate(scoring)*100
print("accuracy: {}%" .format(round(rate,2)))
