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

def select_where(sqlContext, properties, url: str, table: str, col: str, id) -> pd.DataFrame:
    df = DataFrameReader(sqlContext).jdbc(url=f'jdbc:{url}', table=f'"{table}"', properties=properties)
    return df.filter(df[col] == id)


def s_by_company(url: str, table: str) -> pd.DataFrame:
    return select_where(sqlContext, properties, url, table, '公司代號', '5522')


def s_by_stock(url: str, table: str) -> pd.DataFrame:
    return select_where(sqlContext, properties, url, table, '證券代號', '5522')


def toPandas(df, cols=[]):
    global dfp
    if cols != []:
        dfp = df.toPandas()[cols]
    else:
        dfp = df.toPandas()

def rename(d, df):
    for old, new in d.items():
        df = df.withColumnRenamed(old, new)
    return df


def drop(cols, df):
    for col in cols:
        df = df.drop(col)
    return df


def join(x, y):
    return x.join(y, on=[col for col in x.columns if col in y.columns], how='outer')

#--- mops ---
mops = f'postgresql://localhost:{PG_PORT}/mops'

# def de_accumulation(partitionby, cols_exclude, df):
#     windowSpec = Window.partitionBy(partitionby).orderBy(['季'])
#     cols = filter(lambda x : x not in cols_exclude, df.columns)
#     for col in cols:
#         df1 = df.filter(df['季'] == '1').withColumn(col, df[col])
#         df2 = df.withColumn(col, df[col] - func.lag(df[col]).over(windowSpec))
#         df2 = df2.filter(df2[col] != '1')
#         df = df1.union(df2)
#     return df.sort(partitionby + ['季'])


def de_accumulation(partitionby, cols_exclude, df):
    windowSpec = Window.partitionBy(partitionby).orderBy(['季'])
    cols = filter(lambda x : x not in cols_exclude, df.columns)
    for col in cols:
        df = df.withColumn(col, func.when(df['季'] != '1', df[col] - func.lag(df[col]).over(windowSpec)).otherwise(df[col]))
    return df.sort(partitionby + ['季'])


inc = s_by_company(mops, 'ifrs前後-綜合損益表')
inc = de_accumulation(['公司代號', '年'], ['年', '季', '公司代號', '公司名稱'], inc)


# percentage change
def grow(window, cols, n, df):
    for col in cols:
        df = df.withColumn(col+':grow', (df[col] - func.lag(df[col], n).over(window))/func.lag(df[col], n).over(window))
    return df


windowSpec = Window.partitionBy(['公司代號']).orderBy(['年', '季'])

inc = grow(windowSpec, ['營業收入'], 1, inc).withColumnRenamed('營業收入:grow', '營業收入:season:grow')

#toPandas(inc, ['公司代號', '年', '季', '營業收入', '營業收入.season.grow'])


def rolling_sum(window, cols, n, df):
    i = -n + 1
    for col in cols:
        df = df.withColumn(col+':rolling_sum', func.sum(df[col]).over(window.rowsBetween(i, 0)))
    return df


def sum_grow(window, cols, n, df):
    for col in cols:
        df = df.withColumn(col, func.sum(df[col]).over(window.rowsBetween(-n + 1, 0)))
        df = df.withColumn(col+':sum:grow', (df[col] - func.lag(df[col], n).over(window))/func.lag(df[col], n).over(window))
    return df

inc = sum_grow(windowSpec, ['營業收入'], -1, inc).withColumnRenamed('營業收入:sum:grow', '營業收入:half_year:grow')
inc = sum_grow(windowSpec, ['營業收入'], -3, inc).withColumnRenamed('營業收入:sum:grow', '營業收入:year:grow')
#inc.show()
#toPandas(inc, ['公司代號', '年', '季', '營業收入', '營業收入.half_year.grow'])


def rolling_sum_grow(window, cols, n, df):
    for col in cols:
        df = df.withColumn(col, func.sum(df[col]).over(window.rowsBetween(-n + 1, 0)))
        df = df.withColumn(col+':rolling_sum:grow', (df[col] - func.lag(df[col], 1).over(window))/func.lag(df[col], 1).over(window))
    return df

inc = rolling_sum_grow(windowSpec, ['營業收入'], -3, inc).withColumnRenamed('營業收入:rolling_sum:grow', '營業收入:year:rolling:grow')


def ma(window, cols, n, df):
    i = -n + 1
    for col in cols:
        df = df.withColumn(f'{col}:ma{n}', func.mean(df[col]).over(window.rowsBetween(i, 0)))
    return df


inc = ma(windowSpec, ['本期綜合損益總額'], 12, inc)
inc = inc.withColumn('毛利率', inc['營業毛利（毛損）']/inc['營業收入'])
inc = inc.withColumn('營業利益率', inc['營業利益（損失）']/inc['營業收入'])
inc = inc.withColumn('綜合稅後純益率', inc['綜合損益總額歸屬於母公司業主']/inc['營業收入'])

bal = s_by_company(mops, 'ifrs前後-資產負債表-一般業')
bal = drop(['公司名稱', 'Unnamed: 21', '待註銷股本股數（單位：股）', 'Unnamed: 22'], bal)

report = join(inc, bal)
report = report.withColumn('流動比率', report['流動資產']/report['流動負債'])
report = report.withColumn('負債佔資產比率', report['負債總額']/report['資產總額'])
report = report.withColumn('權益報酬率', report['綜合損益總額歸屬於母公司業主'] * 2/(report['權益總額'] + func.lag(report['權益總額'], 4).over(windowSpec)))
report = report.withColumn('profitbility', report['綜合損益總額歸屬於母公司業主'] /func.lag(report['權益總額'], 4).over(windowSpec))
report = grow(windowSpec, ['權益總額'], 4, report).withColumnRenamed('權益總額.grow', 'investment')
report = report.withColumnRenamed('公司代號', '證券代號')
#report.show()

#--- summary ---
summary = f'postgresql://localhost:{PG_PORT}/summary'
ac = s_by_company(summary, '會計師查核報告')
ac = drop(['公司簡稱', '簽證會計師事務所名稱', '簽證會計師','簽證會計師.1', '核閱或查核報告類型'], ac)
ac = rename({'公司代號': '證券代號', '核閱或查核日期': '年月日'}, ac).orderBy(['年', '季', '證券代號'])
report = join(ac, report)

#--- tse ---
tse = f'postgresql://localhost:{PG_PORT}/tse'
close = s_by_stock(tse, '每日收盤行情(全部(不含權證、牛熊證))')
value = s_by_stock(tse, '個股日本益比、殖利率及股價淨值比').drop('證券名稱')
margin = s_by_stock(tse, '當日融券賣出與借券賣出成交量值(元)')
ins = s_by_stock(tse, '三大法人買賣超日報')
fore = s_by_stock(tse, '外資及陸資買賣超彙總表 (股)').drop('證券名稱')
fore = rename({'買進股數':'外資買進股數','賣出股數':'外資賣出股數','買賣超股數':'外資買賣超股數','鉅額交易': '外資鉅額交易'}, fore)
trust = s_by_stock(tse, '投信買賣超彙總表 (股)').drop('證券名稱')
trust = rename({'買進股數':'投信買進股數','賣出股數':'投信賣出股數','買賣超股數':'投信買賣超股數','鉅額交易': '投信鉅額交易'}, trust)
index = DataFrameReader(sqlContext).jdbc(url=f'jdbc:{tse}', table='"大盤統計資訊-收盤指數"', properties=properties)
indexp = DataFrameReader(sqlContext).jdbc(url=f'jdbc:{tse}', table='"大盤統計資訊-漲跌百分比"', properties=properties)
indexp = rename(cytoolz.merge([{col: col + '-漲跌百分比'} for col in indexp.columns if col != '年月日']), indexp)
xdr = s_by_stock(tse, '除權息計算結果表')
m = cytoolz.reduce(join, [close, value, fore, trust, index, indexp, report, xdr])


# set value to privious one if null, column name can not contain '.'
def fill(window, cols, df):
    for col in cols:
        df = df.withColumn(col, func.when(df[col].isNull(), func.lag(df[col]).over(window)).otherwise(df[col]))
    return df

window_stock_day = Window.partitionBy(['證券代號']).orderBy(['年月日'])
m = fill(window_stock_day, report.columns, m).drop('財報年/季')

m = m.withColumn('adj', func.when(m['權值+息值'].isNotNull(), m['權值+息值']).otherwise(0))
m = m.withColumn('adjcum', func.sum('adj').over(window_stock_day.rowsBetween(Window.unboundedPreceding, 0)))
m = m.withColumn('調整收盤價', m['收盤價']+m['adjcum'])
m = m.withColumn('調整開盤價', m['開盤價']+m['adjcum'])
m = m.withColumn('調整最高價', m['最高價']+m['adjcum'])
m = m.withColumn('調整最低價', m['最低價']+m['adjcum'])
m = drop(['adj', 'adjcum'], m)
m = m.withColumn('earning', func.when(m['本益比'] == 0, m['收盤價']/m['本益比']).otherwise(0))
m = m.withColumn('lnmo', func.log(m['調整收盤價']/ (func.lag(m['調整收盤價'], 120).over(window_stock_day))))

def Return(window, cols, periods, df):
    for col in cols:
        for days in periods:
            n = 240/days
            df = df.withColumn(f'return:{days}days:{col}', (func.lead(df[col], days).over(window) / df[col])**n-1)
    return df


m = Return(window_stock_day, ['調整收盤價'], [5, 10, 20, 40, 60, 120], m)    


# log return
def lnReturn(window, cols, periods, df):
    for col in cols:
        for days in periods:
            n = 240/days
            df = df.withColumn(f'logReturn:{days}days:{col}', func.log(func.lead(df[col], days).over(window) / df[col])*n)
    return df


m = lnReturn(window_stock_day, ['調整收盤價'], [5, 10, 20, 40, 60, 120], m)            


def Normalize(window, cols, df):
    for col in cols:
        df = df.withColumn(f'{col}:nmz', (df[col] - func.mean(df[col]).over(window)) / (func.stddev(df[col]).over(window)))
    return df


# this formula is different from pandas because pandas use adjust=true in ewm
def rsi(window, cols, df):
    for col in cols:
        df = df.withColumn('ch', df[col] - func.lag(df[col]).over(window))
        df_tmp = df.filter(df['ch'] < 0)
        df_tmp = df_tmp.withColumn('ch_up', func.when(df_tmp['ch'] < 0, 0).otherwise(0))
        df = df.filter((df['ch'] >= 0) | (df['ch'].isNull()))
        df = df.withColumn('ch_up', df['ch']).union(df_tmp)
        df_tmp = df.filter(df['ch'] > 0)
        df_tmp = df_tmp.withColumn('ch_dn', func.when(df_tmp['ch'] > 0, 0).otherwise(0))
        df = df.filter((df['ch'] <= 0) | (df['ch'].isNull()))
        df = df.withColumn('ch_dn', df['ch']).union(df_tmp)


        df = df.withColumn('upavg', df['ch_up'])
        df = df.withColumn('dnavg', df['ch_dn'])
        df = df.withColumn('upavg', 13/14*func.lag(df['upavg']).over(window) + 1/14*df['ch_up'])
        df = df.withColumn('dnavg', 13/14*func.lag(df['dnavg']).over(window) + 1/14*df['ch_dn'])
        df = df.withColumn(f'rsi:{col}', 100*df['upavg']/(df['upavg'] + df['dnavg']))
    df = drop(['ch', 'ch_up', 'ch_dn'], df)
    return df

m = rsi(window_stock_day, ['調整收盤價'], m)
m.select('rsi:調整收盤價').take(20)

#--- bic ---
bi = f'postgresql://localhost:{PG_PORT}/bic'
bic = DataFrameReader(sqlContext).jdbc(url=f'jdbc:{bi}', table='"景氣指標及燈號-指標構成項目"', properties=properties).drop('年月')
split_col = func.split(m['年月日'], '-')
m = m.withColumn('年', split_col.getItem(0))
m = m.withColumn('月', split_col.getItem(1))
m = join(m, bic)
m = drop(['年', '月'], m)


 
# rsi
m['ch'] = m['調整收盤價'].diff()
m['ch_u'], m['ch_d'] = m['ch'], m['ch']
m.ix[m.ch_u < 0, 'ch_u'] = 0
m.ix[m.ch_d > 0, 'ch_d']= 0
m['ch_d'] = m['ch_d'].abs()
m['rsi'] = m.ch_u.ewm(alpha=1/14).mean()/(m.ch_u.ewm(alpha=1/14).mean()+m.ch_d.ewm(alpha=1/14).mean())*100 #與r和凱基同,ema的公式與一般的ema不同。公式見http://www.fmlabs.com/reference/default.htm?url=RSI.htm
m = m.drop(['ch', 'ch_u', 'ch_d'], axis=1)

def select(sqlContext, properties, url: str, table: str) -> pd.DataFrame:
    df = DataFrameReader(sqlContext).jdbc(url=f'jdbc:{url}', table=f'"{table}"', properties=properties)
    return df
mysum = f'postgresql://localhost:{PG_PORT}/mysum'
df = select(sqlContext, properties, mysum, 'test')
df.show()
df.write.jdbc(url=f'jdbc:{mysum}', table='test', mode='append', properties=properties)
#df = df.withColumn('new', df['name'])

df = pd.DataFrame({'B': [np.nan, 0, 1, 2, 4]})
df = pd.DataFrame({'B': [np.nan, 0, 1]})
df
df.ewm(alpha=1/14).mean()
1/(1+13/14)
1/14*1+13/14*0
1/14*2+13/14*0.07142857142857142

df.filter(df.公司代號 == '5522').show()
windowSpec = Window.partitionBy(['公司代號', '年']).orderBy(['季'])
df.filter(df['公司代號'] == '5522').show()
df.filter(df['季'] == '1').show()
cols = list(filter(lambda x : x not in ['年', '季', '公司代號', '公司名稱'], df.columns))
df1 = df.filter(df['季'] == '1').withColumn('營業收入', df['營業收入'])
df2 = df.withColumn("營業收入", df['營業收入'] - func.lag(df['營業收入']).over(windowSpec))
df2 = df2.filter(df2['營業收入'] != '1')
df1.union(df2).sort(['公司代號', '年', '季']).show()
dfp = df1.union(df2).sort(['公司代號', '年', '季']).toPandas()
dfp[dfp.季 == '2']

data = \
  [("Thin", "Cell Phone", 6000),
  ("Normal", "Tablet", 1500),
  ("Mini", "Tablet", 5500),
  ("Ultra thin", "Cell Phone", 5500),
  ("Very thin", "Cell Phone", 6000),
  ("Big", "Tablet", 2500),
  ("Bendable", "Cell Phone", 3000),
  ("Foldable", "Cell Phone", 3000),
  ("Pro", "Tablet", 4500),
  ("Pro2", "Tablet", 6500)]
df = sqlContext.createDataFrame(data, ["product", "category", "revenue"])
df.show()


# Window function partioned by Category and ordered by Revenue
windowSpec = Window.partitionBy(df['category']).orderBy(df['revenue'].desc())

df['revenue'].over(windowSpec).show()
    
revenue_difference = func.max(df['revenue']).over(windowSpec) - df['revenue']
df.select(
  df['product'],
  df['category'],
  df['revenue'],
  revenue_difference.alias("revenue_difference")).show()

window_over_A = Window.partitionBy('category').orderBy('revenue')
df1 = df.filter(df.revenue == 1500).withColumn("diff", df['revenue'])
df2 = df.withColumn("diff", df['revenue'] - func.lag("revenue").over(window_over_A)).filter(df.revenue != 1500)
df1.union(df2).show()

