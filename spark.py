# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import cytoolz.curried
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, SparkSession, Row, DataFrameReader, DataFrameWriter
from pyspark.sql.window import Window
import pyspark.sql.functions as func
import os
import sys
if os.getenv('MY_PYTHON_PKG') not in sys.path:
    sys.path.append(os.getenv('MY_PYTHON_PKG'))
import syspath
from common.env import PG_PWD, PG_PORT, PG_USER
from common.connection import conn_local_pg
from sql.pg import select

# read from postgres
os.environ['SPARK_CLASSPATH'] = "/home/david/Downloads/postgresql-42.2.1.jar"
sparkClassPath = os.getenv('SPARK_CLASSPATH')
    
# Populate configuration
conf = SparkConf()
#conf.setAppName('application')
conf.setAppName('application').setMaster('local[*]')
#conf.setAppName('application').setMaster('spark://localhost:7077')
conf.set('spark.jars', 'file:%s' % sparkClassPath)
conf.set('spark.executor.extraClassPath', sparkClassPath)
conf.set('spark.driver.extraClassPath', sparkClassPath)
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
properties = {'user':PG_USER, 'password':PG_PWD,"driver": "org.postgresql.Driver"}

def join(x, y):
    return x.join(y, on=[col for col in x.columns if col in y.columns], how='outer')


def drop(cols, df):
    for col in cols:
        df = df.drop(col)
    return df

#--- tse ---
tse = f'jdbc:postgresql://localhost:{PG_PORT}/tse'

close = DataFrameReader(sqlContext).jdbc(url=tse, table='"每日收盤行情(全部(不含權證、牛熊證))"', properties=properties)

close.printSchema()
#close.show(truncate = False)
close.createOrReplaceTempView("close")

close = sqlContext.sql(f'SELECT "年月日", "證券代號", "開盤價", "最高價", "最低價", "收盤價" FROM close')
close.take(10)

xdr = DataFrameReader(sqlContext).jdbc(url=tse, table='"除權息計算結果表"', properties=properties)
xdr.printSchema()
xdr.createOrReplaceTempView("xdr")
xdr = sqlContext.sql(f'SELECT "年月日", "證券代號", "權值+息值" FROM xdr')
xdr.show(10)
xdr.take(10)

df = join(close, xdr)
df.printSchema()
df.show(10)

df1 = close.toPandas()
df2 = xdr.toPandas()


df = sqlContext.createDataFrame(df)


window_stock_day = Window.partitionBy(['證券代號']).orderBy(['年月日'])
df = df.withColumn('adj', func.when(df['權值+息值'].isNotNull(), df['權值+息值']).otherwise(0))
df = df.withColumn('adjcum', func.sum('adj').over(window_stock_day.rowsBetween(Window.unboundedPreceding, 0)))
df = df.withColumn('調整收盤價', df['收盤價']+df['adjcum'])
df = df.withColumn('調整開盤價', df['開盤價']+df['adjcum'])
df = df.withColumn('調整最高價', df['最高價']+df['adjcum'])
df = df.withColumn('調整最低價', df['最低價']+df['adjcum'])
df = drop(['adj', 'adjcum'], df)
df.take(10)

mysum = f'jdbc:postgresql://localhost:{PG_PORT}/mysum'
df.write.jdbc(url=mysum, table='price', mode='append', properties=properties)

#--- summary ---
summary = f'jdbc:postgresql://localhost:{PG_PORT}/summary'
ac = DataFrameReader(sqlContext).jdbc(url=summary, table='"會計師查核報告"', properties=properties)
ac.printSchema()
ac = rename({'公司代號': '證券代號', '核閱或查核日期': '年月日'}, ac).orderBy(['年', '季', '證券代號'])