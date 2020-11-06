from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, col, udf
from pyspark.sql.types import IntegerType
from pyspark.ml.linalg import Vectors,VectorUDT #in dataframe ml.linalg is used
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans,BisectingKMeans,GaussianMixture
import numpy as np


spark = SparkSession.Builder().master('local').appName('twitter').getOrCreate()
sc = spark.sparkContext

train_df = spark.read.csv('/home/luminar/Downloads/twitter datas/train.csv',header=True, inferSchema=True)
train_df.show()
test_df = spark.read.csv('/home/luminar/Downloads/twitter datas/test.csv',header=True, inferSchema=True)
test_df.show()

train_df.filter(col('_c11').isNotNull()).show(truncate=False)

adc=[c for c in train_df.columns if c.startswith("_") or c.endswith("Text")]
print(adc)
print(*adc)


train_df=train_df.fillna('').withColumn("ST", concat(*adc))
train_df.show()

train_df.filter(col('ItemID')==9481).show(truncate=False)
train_df.select('ST').show(truncate=False)

f1=open("/home/luminar/Downloads/twitter datas/positive-words.txt")
f2=open("/home/luminar/Downloads/twitter datas/negative-words.txt")
# positive=[x.split('\n')[0] for x in f1.readlines()]
positive=[x.strip() for x in f1.readlines()]
negative=[x.strip() for x in f2.readlines()]
# print(positive)
# print(negative)

def to_values(x):
    p=0
    n=0
    for w in x.split():
        if w in positive:
            p+=1
        if w in negative:
            n+=1
    return Vectors.dense(np.array([p,n]))
to_values_udf=udf(lambda x:to_values(x),VectorUDT())#vectorudt is used to convert string into vectors
train_df=train_df.withColumn("features",to_values_udf(col("ST")))
train_df.show()

test_df=test_df.withColumn("features",to_values_udf(col("SentimentText")))
test_df.show()

#two Columns
# def tp_values(x, i):
#     k = 0
#     for w in x.split():
#         if i == 0:
#             if w in positive:
#                 k += 1
#         else:
#             if w in negative:
#                 k += 1
#     return k
#
#
# positive_udf = udf(lambda x: tp_values(x, 0))
# negative_udf = udf(lambda x: tp_values(x, 1))
# #
#
# #
# train_df = train_df.withColumn('pos', positive_udf(col('ST')).astype(IntegerType())) \
#     .withColumn('neg', negative_udf(col('ST')).astype(IntegerType()))
#
# train_df.show()
#
# assembler = VectorAssembler(inputCols=['pos', 'neg'], outputCol='features')
# train_df = assembler.transform(train_df)
# train_df.show()


#modelling


kmeans=KMeans().setK(2).setSeed(1).setMaxIter(20)
model=kmeans.fit(train_df)
model.transform(test_df).show()
for c in model.clusterCenters():
    print (c)
#



bkmeans=BisectingKMeans().setK(2).setSeed(1).setMaxIter(20)
model=bkmeans.fit(train_df)
model.transform(test_df).show()
for c in model.clusterCenters():
    print (c)



gaussianmixture=GaussianMixture().setK(2).setSeed(1)
model=gaussianmixture.fit(train_df)
model.transform(test_df).show()