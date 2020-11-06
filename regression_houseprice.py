from pyspark.sql import SparkSession
from pyspark.sql.functions import col,when
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer,OneHotEncoder,VectorAssembler
from pyspark.ml.regression import LinearRegression

spark=SparkSession.Builder().master("local").appName("houseprice").getOrCreate()
sc=spark.sparkContext

train_df=spark.read.csv("/home/luminar/Downloads/train(1).csv",header=True,inferSchema=True)
train_df.show()

train_df.printSchema()
print(train_df.count())

#to find missing values
for c in train_df.columns:
    print(c,train_df.filter(col(c).isNull()).count())

for c in train_df.columns:
    print(c,train_df.filter(col(c)=="NA").count())

#to drop the columns with missig value greater than 1000
for c in train_df.columns:
    if train_df.filter(col(c)=="NA").count() >1000:
        train_df.drop(c)

#MAY 4TH
#fill the null values of columns by using when and 0therwise


train_df.groupBy("LotFrontage").count().show()
train_df=train_df.withColumn("LotFrontage",when(col("LotFrontage")=="NA",0).otherwise(col("LotFrontage")))
train_df=train_df.withColumn("LotFrontage",when(col("LotFrontage")=="NA",0).otherwise(col("LotFrontage")).cast(IntegerType()))

train_df.groupBy("MasVnrType").count().show()
train_df=train_df.withColumn("MasVnrType",when(col("MasVnrType")=="NA","None").otherwise(col("MasVnrType")))

train_df.groupBy("MasVnrArea").count().show()
train_df=train_df.withColumn("MasVnrArea",when(col("MasVnrArea")=="NA",0).otherwise(col("MasVnrArea")).cast(IntegerType()))

mode=train_df.groupBy("Electrical").count().show()
mode_elec=train_df.groupBy("Electrical").count().orderBy("count",ascending=False).first()[0]

train_df=train_df.withColumn("Electrical",when(col("Electrical")=="NA",mode_elec).otherwise(col("Electrical")))

for c in train_df.columns:
    print(c,train_df.filter(col(c)=="NA").count())


print(train_df.dtypes)


col_names=[c[0] for c in train_df.dtypes if c[1]=='string']
#convert all categorical columns  numerical columns using strinindexer with column name col_name+index
def string_index_fun(col_name):
    stringindexer= StringIndexer(inputCol=col_name,outputCol=col_name+'_Index')
    model=stringindexer.fit(train_df)
    indexed=model.transform(train_df)
    # return indexed
#convert all index columns into vector columns using onehotencoder with column name col_name+vector
    encoder=OneHotEncoder(inputCol=col_name+'_Index',outputCol=col_name+'_Vector')
    encoded=encoder.transform(indexed)
    return encoded

for c in col_names:
    train_df=string_index_fun(c)

train_df.show()

for c in train_df.columns:
    if c=='Id' or c.endswith('Index') or c in col_names:
        train_df=train_df.drop(c)

train_df.show()

inputcols=train_df.columns
inputcols.remove('SalePrice')
print(inputcols)

assembler=VectorAssembler(inputCols=inputcols,outputCol='features')
train_df=assembler.transform(train_df)
train_df.show(truncate=False)



#modelling split the data into train and test
train,test=train_df.randomSplit([0.8,0.2],seed=1)


#fit the train data
lr=LinearRegression(labelCol='SalePrice',featuresCol='features')
model=lr.fit(train)

#predict the data using test data
predict_test=model.transform(test)
predict_test.select("SalePrice","prediction").show()

print(model.summary.rootMeanSquaredError)
print(model.summary.r2)


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import DecisionTreeRegressor
dt=DecisionTreeRegressor(labelCol='SalePrice',featuresCol='features')
dtmodel=dt.fit(train)
predict_dt=dtmodel.transform(test)
predict_dt.select("SalePrice","prediction").show()

re=RegressionEvaluator(labelCol="SalePrice",predictionCol="prediction",metricName="r2")
r2=re.evaluate(predict_dt)
print(r2)


from pyspark.ml.regression import RandomForestRegressor
rf=RandomForestRegressor(labelCol="SalePrice",featuresCol="features")
rfmodel=rf.fit(train)
predict_rf=rfmodel.transform(test)
predict_rf.select("SalePrice","prediction").show()

r2=re.evaluate(predict_rf)
print(r2)