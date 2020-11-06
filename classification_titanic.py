import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, col, udf,mean
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer,OneHotEncoder,VectorAssembler
from pyspark.ml.classification import LogisticRegression
spark = SparkSession.Builder().master('local').appName('titanic').getOrCreate()
sc = spark.sparkContext

#readthe dataframe
titanic_df = spark.read.csv('/home/luminar/Downloads/titanic.csv',header=True, inferSchema=True)
titanic_df.show(100)

#print the datatypes
titanic_df.printSchema()


#correlation b/w pclass and survived
print(titanic_df.stat.corr("Pclass","Age"))

#correlation b/w all columns with survived
for c in titanic_df.columns:
    if(titanic_df.select(c).dtypes[0][1]!="string"):
        print(c+'='+str(titanic_df.stat.corr(c,"Survived")))

#to count the missing values
for c in titanic_df.columns:
    print(c,titanic_df.filter(col(c).isNull()).count())

#to find the mode of embarked column

titanic_df.groupBy("Embarked").count().orderBy("count",ascending=False).show()
mode_embarked=titanic_df.groupBy("Embarked").count().orderBy("count",ascending=False).first()[0]
print(mode_embarked)


#to fill the nullvales of embarked column with it.s mode
# titanic_df=titanic_df.fillna(mode_embarked,subset=['Embarked'])

#to count the missing values
# for c in titanic_df.columns:
#     print(c,titanic_df.filter(col(c).isNull()).count())


#to find the mean of age column
titanic_df.select(mean('Age')).show()
mean_age=titanic_df.select(mean('Age')).first()[0]
print(int(mean_age))

#to fill the null values of age column with mean of age
# titanic_df=titanic_df.fillna(mean_age,subset=['Age'])

#to count the missing values
# for c in titanic_df.columns:
#     print(c,titanic_df.filter(col(c).isNull()).count())

#APRIL 29TH
#0r
titanic_df=titanic_df.fillna({"Age":int(mean_age),"Embarked":mode_embarked})
for c in titanic_df.columns:
    print(c,titanic_df.filter(col(c).isNull()).count())

#drop the cabin column
titanic_df=titanic_df.drop("Cabin")
titanic_df.show()

#extract the titles in name and find it's count
def title_extract(name):
    return re.findall(r'( [A-Za-z]+)\.',name)[0]

title_extract_udf=udf(title_extract)
titanic_df=titanic_df.withColumn("Title",title_extract_udf("Name"))
titanic_df.groupBy("Title").count().show()

#convert the categorical variables to numerical variables(in machine learning=label encoding in pysprk=stringindexer)

string1=StringIndexer(inputCol='Sex',outputCol='sex_index')
model=string1.fit(titanic_df)
index_df=model.transform(titanic_df)
index_df.show()



#convert all categorical columns  numerical columns using strinindexer with column name col_name+index
def string_index_fun(col_name):
    stringindexer= StringIndexer(inputCol=col_name,outputCol=col_name+'_Index')
    model=stringindexer.fit(titanic_df)
    indexed=model.transform(titanic_df)
    # return indexed
#convert all index columns into vector columns using onehotencoder with column name col_name+vector
    encoder=OneHotEncoder(inputCol=col_name+'_Index',outputCol=col_name+'_Vector')
    encoded=encoder.transform(indexed)
    return encoded

indextables = ['Embarked','Sex','Title']

for c in indextables:
    titanic_df=string_index_fun(c)

titanic_df.show()


#drop all the object columns and indexcolumn
titanic_df=titanic_df.drop('PassengerId','Title','Embarked','Name','Ticket','Sex')
titanic_df=titanic_df.drop(*[c for c in titanic_df.columns if c.endswith('Index')])
titanic_df.show()



inputcols=titanic_df.columns
inputcols.remove('Survived')
print(inputcols)

#combined all vectrocolumns into a feature column using vectorassembler
assembler=VectorAssembler(inputCols=inputcols,outputCol='features')
titanic_df=assembler.transform(titanic_df)
titanic_df.show(truncate=False)

#modelling split the data into train and test
train,test=titanic_df.randomSplit([0.8,0.2],seed=1)
print(train.count(),test.count())

#fit the train data
lr=LogisticRegression(labelCol='Survived',featuresCol='features')
model=lr.fit(train)

#predict the data using test data
predict_test=model.transform(test)
predict_test.select("survived","prediction").show()

#calculate the accuracy
tp=predict_test.filter((col("survived")==0)&(col("prediction")==0)).count()
tn=predict_test.filter((col("survived")==1)&(col("prediction")==1)).count()
fp=predict_test.filter((col("survived")==1)&(col("prediction")==0)).count()
fn=predict_test.filter((col("survived")==0)&(col("prediction")==1)).count()
print(tp,tn,fp,fn)

print("acc=",(tp+tn)/(tp+tn+fp+fn))

#APRIL30TH
#print coefficients and intercepts
print(model.coefficients,model.intercept)
training_summary = model.summary

print("false positive rate by label:")
for i,rate in enumerate(training_summary.falsePositiveRateByLabel):
    print("label %d :%s" % (i,rate))

print("true positive rate by label:")
for i,rate in enumerate(training_summary.truePositiveRateByLabel):
    print("label %d :%s" % (i,rate))

print("precision by label:")
for i,prec in enumerate(training_summary.precisionByLabel):
    print("label %d :%s" % (i,prec))

print("recall by label:")
for i,rec in enumerate(training_summary.recallByLabel):
    print("label %d :%s" % (i,rec))

print("fmeasure by label:")
for i,f in enumerate(training_summary.fMeasureByLabel()):
    print("label %d :%s" % (i,f))


accuracy=training_summary.accuracy
print("acc=",accuracy)


#MAY 4TH
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator=MulticlassClassificationEvaluator(labelCol="Survived",predictionCol="prediction",metricName="accuracy")
accuracy=evaluator.evaluate(predict_test)
print("test error= %g" % (1.0-accuracy))
print(accuracy)


from pyspark.ml.classification import DecisionTreeClassifier
dt=DecisionTreeClassifier(labelCol="Survived",featuresCol="features")
dtmodel=dt.fit(train)
predict_train=dtmodel.transform(train)
predict_test=dtmodel.transform(test)
predict_test.select("survived","prediction").show()

tp=predict_test.filter((col("survived")==0)&(col("prediction")==0)).count()
tn=predict_test.filter((col("survived")==1)&(col("prediction")==1)).count()
fp=predict_test.filter((col("survived")==1)&(col("prediction")==0)).count()
fn=predict_test.filter((col("survived")==0)&(col("prediction")==1)).count()
print(tp,tn,fp,fn)

print("acc=",(tp+tn)/(tp+tn+fp+fn))


from pyspark.ml.classification import RandomForestClassifier
dt=RandomForestClassifier(labelCol="Survived",featuresCol="features")
dtmodel=dt.fit(train)
predict_test=dtmodel.transform(test)
predict_test.select("survived","prediction").show()

tp=predict_test.filter((col("survived")==0)&(col("prediction")==0)).count()
tn=predict_test.filter((col("survived")==1)&(col("prediction")==1)).count()
fp=predict_test.filter((col("survived")==1)&(col("prediction")==0)).count()
fn=predict_test.filter((col("survived")==0)&(col("prediction")==1)).count()
print(tp,tn,fp,fn)

print("acc=",(tp+tn)/(tp+tn+fp+fn))