import inline as inline
from matplotlib import pyplot as plt
from pyspark.pandas.plot import matplotlib
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, StructType, StructField, DateType, IntegerType, DoubleType
from pyspark.sql.functions import col, expr, year, month

spark = (SparkSession.builder.appName("AppleStockDataAnalysis").getOrCreate())
source_data_path_csv = "../data/AAPL.csv"

'''
CSV file
'''
print("Before applying the custom schema to the file")
apple_df = (
    spark.read.format("csv")
        .option("header", True)
        .load(source_data_path_csv))

apple_df.show()

print(apple_df.printSchema())

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

apple_df1 = (
    spark.read.format("csv")
        .option("header", True)
        .schema(schema)
        .load(source_data_path_csv)
)

apple_df1.show()
print(apple_df1.printSchema())

'''
Working with Pyspark Dataframe
'''
print("Some Dataframe columns Manipulations")
import pyspark.sql.functions as f

df = apple_df1.withColumn('date', f.to_date('Date'))
print(df.show())

# Manipulating the dataframe
print("Date Breakdown")
date_breakdown = ['year', 'month', 'day']

for i in enumerate(date_breakdown):
    index = i[0]
    name = i[1]
    df = df.withColumn(name, f.split('date', '-')[index])

print(df.show())

# Showing the data visualization
df_plot = df.select('year', 'Adj Close').toPandas()

# from matplotlib import pyplot as plt
# # %matplotlib inline
# df_plot.set_index('year', inplace=True)
# df_plot.plot(figsize=(16, 6), grid=True)
# plt.title('Apple stock')
# plt.ylabel('Stock Quote ($)')
# plt.show()

print(df.toPandas().shape)

print(df.dropna().count())
'''Describing the data'''
df.select('Open', 'High', 'Low', 'Close', 'Adj Close').describe().show()

'''GroupBy on the row of the data'''

df.groupBy(['year']).agg({'Adj Close': 'count'}).withColumnRenamed('count(Adj Close)', 'Row Count').orderBy(["year"],
                                                                                                            ascending=False).show()

'''Applying the Machine Learning'''

trainDF = df[df.year < 2022]
testDF = df[df.year > 2021]

print(trainDF.toPandas().shape)

print(testDF.toPandas().shape)

trainDF_plot = trainDF.select('year', 'Adj Close').toPandas()
trainDF_plot.set_index('year', inplace=True)
trainDF_plot.plot(figsize=(16, 6), grid=True)
plt.title('Apple Stock start-2022')
plt.ylabel('Stock Quote ($)')

testDF_plot = testDF.select('year', 'Adj Close').toPandas()
testDF_plot.set_index('year', inplace=True)
testDF_plot.plot(figsize=(16, 6), grid=True)
plt.title('Apple Stock 2021-2022')
plt.ylabel('Stock Quote ($)')

import numpy as np

print("Changing into the numpy")
trainArray = np.array(trainDF.select('Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close').collect())
testArray = np.array(testDF.select('Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close').collect())

print(trainArray[0])
print('---------------')
print(testArray[0])

from sklearn.preprocessing import MinMaxScaler

minMaxScale = MinMaxScaler()

minMaxScale.fit(trainArray)

testingArray = minMaxScale.transform(testArray)
trainingArray = minMaxScale.transform(trainArray)

print(testingArray[0])
print('---------------')
print(trainingArray[0])

xtrain = trainingArray[:, 0:-1]
xtest = testingArray[:, 0:-1]
ytrain = trainingArray[:, -1:]
ytest = testingArray[:, -1:]

print(trainingArray[0])

print(xtrain[0])

print(ytrain[0])

print('xtrain shape = {}'.format(xtrain.shape))
print('xtest shape = {}'.format(xtest.shape))
print('ytrain shape = {}'.format(ytrain.shape))
print('ytest shape = {}'.format(ytest.shape))

plt.figure(figsize=(16, 6))
plt.plot(xtrain[:, 0], color='red', label='open')
plt.plot(xtrain[:, 1], color='blue', label='high')
plt.plot(xtrain[:, 2], color='green', label='low')
plt.plot(xtrain[:, 3], color='purple', label='close')
plt.legend(loc='upper left')
plt.title('Open, High, Low, and Close by Day')
plt.xlabel('Days')
plt.ylabel('Scaled Quotes')

plt.figure(figsize=(16, 6))
plt.plot(xtrain[:, 4], color='black', label='volume')
plt.legend(loc='upper right')
plt.title('Volume by Day')
plt.xlabel('Days')
plt.ylabel('Scaled Volume')
# plt.show()

from keras import models, layers

model = models.Sequential()
model.add(layers.LSTM(1, input_shape=(1, 5)))
model.add(layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

xtrain = xtrain.reshape((xtrain.shape[0], 1, xtrain.shape[1]))
xtest = xtest.reshape((xtest.shape[0], 1, xtest.shape[1]))

print('The shape of xtrain is {}: '.format(xtrain.shape))
print('The shape of xtest is {}: '.format(xtest.shape))

loss = model.fit(xtrain, ytrain, batch_size=10, epochs=100)
print(loss)

plt.plot(loss.history['loss'], label='loss')
plt.title('mean squared error by epoch')
plt.legend()
plt.show()

# predicted = model.predict(xtest)

plt.plot(loss.history['loss'], label='loss')
plt.title('mean squared error by epoch')
plt.legend()
plt.show()

predicted = model.predict(xtest)
combined_array = np.concatenate((ytest, predicted), axis=1)
plt.figure(figsize=(16, 6))
plt.plot(combined_array[:, 0], color='red', label='actual')
plt.plot(combined_array[:, 1], color='blue', label='predicted')
plt.legend(loc='lower right')
plt.title('2022 Actual vs. Predicted Apple Stock')
plt.xlabel('Days')
plt.ylabel('Scaled Quotes')
plt.show()

import sklearn.metrics as metrics

print(np.sqrt(metrics.mean_squared_error(ytest, predicted)))
