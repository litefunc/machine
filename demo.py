# demo.py is old version, mysum.py is new version

import os
import time
import functools
from datetime import datetime, timedelta
from copy import deepcopy
import cytoolz.curried
import pandas as pd
import numpy as np
import os
import sys

if os.getenv('MY_PYTHON_PKG') not in sys.path:
    sys.path.np.append(os.getenv('MY_PYTHON_PKG'))

import syspath
from common.connection import conn_local_lite
import sqlCommand as sqlc

start = time.time()

def timeDelta(s):
    global start
    end = time.time()
    print(s,'timedelta: ', end - start)
    start = end


conn_lite = conn_local_lite('mops.sqlite3')
cur_lite = conn_lite.cursor()


# from pandas.io import data, wb
# from pandas_datareader import data, wb
# import pandas.io.data as web
# import pandas_datareader.data as web


def timeSpan(func):
    def wrapper(*args, **kw):
        print('call %s():' % func.__name__)
        start = time.time()
        x = func(*args, **kw)
        end = time.time()
        print('Complete in {} second(s)'.format(end-start))
        return x
    return wrapper

## --- read from sqlite ---
def mymerge(x, y):
    m = pd.merge(x, y, on=[col for col in list(x) if col in list(y)], how='outer')
    return m

# --- report---
def fill(s):
    a = np.array(0)
    r = s[~pd.isnull(s)].index
    a = np.append(a, r)
    a = np.append(a, len(s))
    le = a[1:] - a[:len(a) - 1]
    l = []
    for i in range(len(le)):
        l = l + np.repeat(s[a[i]], le[i]).tolist()
    return pd.Series(l, name=s.name)
id='5522'
companyId = "'{}'".format(id)
sql = "SELECT * FROM '{}' WHERE 公司代號 LIKE {}" .format('ifrs前後-綜合損益表', companyId)
inc = pd.read_sql_query(sql, conn_lite)
inc.dtypes
floatColumns = list(filter(lambda x : x not in ['年', '季', '公司代號', '公司名稱'], list(inc)))
inc[floatColumns] = inc[floatColumns].astype(float)
# def change(df):
#     df0 = df[[x for x in list(df) if df[x].dtype == 'object']]
#     df1 = df[[x for x in list(df) if df[x].dtype != 'object']]
#     a0 = np.array(df0)
#     a1 = np.array(df1)
#     v = np.vstack((a1[0], a1[1:] - a1[0:len(df) - 1]))
#     h = np.hstack((a0, v))
#     return pd.DataFrame(h, columns=list(df0) + list(df1))

def change1(df, columns):
    df0 = df[columns]
    df1 = df[list(filter(lambda x : x not in columns, list(df)))]
    a0 = np.array(df0)
    a1 = np.array(df1)
    v = np.vstack((a1[0], a1[1:] - a1[0:len(df) - 1]))
    h = np.hstack((a0, v))
    return pd.DataFrame(h, columns=list(df0) + list(df1))

# inc = inc.groupby(['公司代號', '年']).apply(change).reset_index(drop=True)  #'季' must be string
inc = inc.groupby(['公司代號', '年']).apply(change1, ['年', '季', '公司代號', '公司名稱']).reset_index(drop=True)  #'季' must be string
inc['grow_s'] = inc['本期綜合損益總額'].pct_change(1)
inc['grow_hy'] = inc['本期綜合損益總額'].rolling(window=2).sum().pct_change(2)
# inc[col1] = inc[col1].rolling(window=4).sum()
inc[floatColumns] = inc[floatColumns].rolling(window=4).sum()
inc['grow_y'] = inc['本期綜合損益總額'].pct_change(4)
inc['grow'] = inc['本期綜合損益總額'].pct_change(1)
# inc['grow.ma'] = inc['grow'].rolling(window=24).mean()*4
inc['本期綜合損益總額.wma'] = inc.本期綜合損益總額.ewm(com=19).mean() * 4
inc['本期綜合損益總額.ma'] = inc['本期綜合損益總額'].rolling(window=12).mean() * 4
sql = "SELECT * FROM '{}' WHERE 公司代號 LIKE {}"
bal = pd.read_sql_query(sql .format('ifrs前後-資產負債表-一般業', companyId), conn_lite)
bal[['年', '季']]=bal[['年', '季']].astype(str)
del bal['公司名稱']
timeDelta('mops')

#--- summary ---
conn_lite = conn_local_lite('summary.sqlite3')
sql = "SELECT * FROM '{}' WHERE 公司代號 LIKE {}" .format('會計師查核報告', companyId)
ac = pd.read_sql_query(sql, conn_lite).rename(columns={'公司代號': '證券代號', '公司簡稱': '證券名稱', '核閱或查核日期': '年月日'}).sort_values(['年', '季', '證券代號']).drop(['簽證會計師事務所名稱', '簽證會計師','簽證會計師.1', '核閱或查核報告類型'], axis=1)
ac[['年', '季']]=ac[['年', '季']].astype(str)
del ac['證券名稱']

# companyId = "'3056%'"
sql = "SELECT * FROM '{}' WHERE 公司代號 LIKE {}"
fin = pd.read_sql_query(sql .format('財務分析', companyId), conn_lite)
del fin['公司簡稱']
report = mymerge(inc, bal)
inc.dtypes
bal.dtypes
report['流動比率'] = report['流動資產'] / report['流動負債']
report['負債佔資產比率'] = report['負債總額'] / report['資產總額']
report['權益報酬率'] = report['綜合損益總額歸屬於母公司業主'] * 2 / (report['權益總額'] + report['權益總額'].shift())
report['profitbility'] = report.綜合損益總額歸屬於母公司業主 / (report.權益總額.shift(4))
report['investment'] = report.權益總額.pct_change(4)
report = report.rename(columns={'公司代號': '證券代號'})
report = mymerge(ac, report)
remcol = ['Unnamed: 21', '待註銷股本股數（單位：股）', 'Unnamed: 22', ]
report = report.drop(remcol, axis=1)
report[['年', '季', '綜合損益總額歸屬於母公司業主', '權益總額', 'profitbility', '權益報酬率']]

timeDelta('summary')

#--- tse ---
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

conn_lite = conn_local_lite('tse.sqlite3')
sql="SELECT * FROM '{}' WHERE 證券代號 LIKE {}"

# read from redis
# close = read_msgpack(r.get("close:{}".format(companyId)))
# value = read_msgpack(r.get("value:{}".format(companyId)))
# margin = read_msgpack(r.get("margin:{}".format(companyId)))
# ins = read_msgpack(r.get("ins:{}".format(companyId)))

# read form sqlite
close = pd.read_sql_query(sql.format('每日收盤行情(全部(不含權證、牛熊證))', companyId), conn_lite)
value = pd.read_sql_query(sql.format('個股日本益比、殖利率及股價淨值比', companyId), conn_lite).drop(['證券名稱'], 1)
margin = pd.read_sql_query(sql.format('當日融券賣出與借券賣出成交量值(元)', companyId), conn_lite)
ins = pd.read_sql_query(sql.format('三大法人買賣超日報', companyId), conn_lite)
# deal = pd.read_sql_query(sql.format('自營商買賣超彙總表 (股)', companyId), conn_lite).drop(['證券名稱'], 1)

# deal = read_msgpack(r.get("deal:{}".format(companyId)))
# deal[['自營商(自行買賣)賣出股數', '自營商(自行買賣)買賣超股數', '自營商(自行買賣)買進股數', '自營商(避險)賣出股數', '自營商(避險)買賣超股數', '自營商(避險)買進股數', '自營商賣出股數', '自營商買賣超股數', '自營商買進股數']] = deal[['自營商(自行買賣)賣出股數', '自營商(自行買賣)買賣超股數', '自營商(自行買賣)買進股數', '自營商(避險)賣出股數', '自營商(避險)買賣超股數', '自營商(避險)買進股數', '自營商賣出股數', '自營商買賣超股數', '自營商買進股數']].fillna(0)

fore = pd.read_sql_query(sql.format('外資及陸資買賣超彙總表 (股)', companyId), conn_lite).drop(['證券名稱'], 1).rename(columns={'買進股數':'外資買進股數','賣出股數':'外資賣出股數','買賣超股數':'外資買賣超股數','鉅額交易': '外資鉅額交易'})
trust = pd.read_sql_query(sql.format('投信買賣超彙總表 (股)', companyId), conn_lite).drop(['證券名稱'], 1).rename(columns={'買進股數':'投信買進股數','賣出股數':'投信賣出股數','買賣超股數':'投信買賣超股數','鉅額交易': '投信鉅額交易'})
index = pd.read_sql_query("SELECT * FROM '{}' WHERE 指數 LIKE {}".format('大盤統計資訊', "'建材營造類指數'"), conn_lite).rename(columns={'收盤指數':'建材營造類指數'}).drop(['指數', '漲跌(+/-)'], axis=1)
index1 = pd.read_sql_query("SELECT * FROM '{}'".format('index'), conn_lite)
rindex = pd.read_sql_query("SELECT * FROM '{}' WHERE 指數 LIKE {}".format('大盤統計資訊', "'建材營造類報酬指數'"), conn_lite).rename(columns={'收盤指數':'建材營造類報酬指數', '漲跌點數':'r漲跌點數','漲跌百分比(%)':'r漲跌百分比(%)'}).drop(['指數', '漲跌(+/-)'], axis=1)
close['本益比'] = close['本益比'].replace('0.00', np.nan) # pe is '0.00' when pe < 0
value['本益比'] = value['本益比'].replace('-', np.nan)  # pe is '-' when pe < 0
value['股價淨值比'] = value['股價淨值比'].replace('-', np.nan)
sql = "SELECT * FROM '{}' WHERE 股票代號 LIKE {}"
xdr = pd.read_sql_query(sql.format('除權息計算結果表', companyId), conn_lite).rename(columns={'股票代號': '證券代號', '股票名稱': '證券名稱'})

timeDelta('tse')

m = cytoolz.reduce(mymerge, [close, value, fore, trust, index, index1, rindex, report, xdr])
timeDelta('merge')
# for df in [close, value, fore, trust, index, index1, rindex, report, xdr]:
#     print('--' in df)
m.dtypes
report.dtypes
m.年月日 = pd.to_datetime(m.年月日, format='%Y/%m/%d').apply(lambda x: x.date()) # should convert to datetime before sort, or the result is  wrong
m=m.sort_values(['年月日','證券代號']).reset_index(drop=True) # reset_index make the index ascending
m[list(report)] = m[list(report)].apply(fill)
m['淨利（淨損）歸屬於母公司業主'] = m['淨利（淨損）歸屬於母公司業主'].astype(float)
m['綜合損益總額歸屬於母公司業主'] = m['綜合損益總額歸屬於母公司業主'].astype(float)
m['毛利率'] = m['營業毛利（毛損）']/m['營業收入']
m['營業利益率'] = m['營業利益（損失）']/m['營業收入']
m['綜合稅後純益率'] = m['綜合損益總額歸屬於母公司業主']/m['營業收入']
m['time'] = m.index.tolist()
col = ['年月日', '證券代號', '年', '季']
m = m.replace('--', np.nan)

del m['財報年/季']
floatColumns = [col for col in list(m) if col not in ['年月日', '證券代號', '年', '季', '證券名稱']]
# for i in floatColumns:
#     print(i)
#     m[i].astype(float)
m[floatColumns]=m[floatColumns].astype(float)

m = m[col+[x for x in list(m) if x not in col]]
col = ['年月日', '證券代號', '證券名稱', '公司名稱', '年', '季', '漲跌(+/-)', '外資鉅額交易', '投信鉅額交易', '財報年/季']
m[[x for x in list(m) if x not in col]] = m[[x for x in list(m) if x not in col]].astype(float)
col = ['年月日', '證券代號', 'time', '成交股數', '成交筆數', '成交金額', '開盤價', '最高價', '最低價', '收盤價', '調整收盤價', '漲跌(+/-)', '漲跌價差', '最後揭示買價', '最後揭示買量', '最後揭示賣價',
  '最後揭示賣量', '本益比', '殖利率(%)', '股價淨值比', '自營商(自行買賣)賣出股數', '自營商(自行買賣)買賣超股數', '自營商(自行買賣)買進股數', '自營商(避險)賣出股數', '自營商(避險)買賣超股數',
  '自營商(避險)買進股數', '自營商賣出股數', '自營商買賣超股數', '自營商買進股數', '外資鉅額交易', '外資買進股數', '外資賣出股數', '外資買賣超股數', '投信鉅額交易', '投信買進股數',
  '投信賣出股數', '投信買賣超股數', '基本每股盈餘（元）','每股參考淨值', '流動比率', '負債佔資產比率', '權益報酬率', '毛利率', '營業利益率', '綜合稅後純益率', 'grow_s', 'grow_hy', 'grow_y', 'grow',
    '本期綜合損益總額.wma', '本期綜合損益總額.ma', 'profitbility', 'investment', '建材營造類指數', '漲跌點數', '漲跌百分比(%)', '建材營造類報酬指數', 'r漲跌點數', 'r漲跌百分比(%)']+list(index1)[1:]
col = [ii for n,ii in enumerate(col) if ii not in col[:n]]

col = [col for col in list(m) if col not in ['公司名稱', '年', '季']]
# m[['profitbility', '權益報酬率']]
timeDelta('before dropna')
list(m)
m['a'] = m['權值+息值'].replace(NaN, 0)
m['b'] = m['a'].cumsum()
m['調整收盤價']=m.收盤價+m.b
m = m.drop(['a', 'b'], axis=1)
m=m.dropna(axis=1, how='all')

# TWII = web.DataReader("^TWII", "yahoo").reset_index()[['Date', 'Adj Close']].rename(columns={'Date': '年月日', 'Adj Close':'TWII'})
# SSE = web.DataReader("000001.SS", "yahoo").reset_index()[['Date', 'Adj Close']].rename(columns={'Date': '年月日', 'Adj Close':'SSE'})
# HSI = web.DataReader("^HSI", "yahoo").reset_index()[['Date', 'Adj Close']].rename(columns={'Date': '年月日', 'Adj Close':'HSI'})
# # STI = web.DataReader("^STI", "yahoo").reset_index()[['Date', 'Adj Close']].rename(columns={'Date': '年月日', 'Adj Close':'STI'})
# N225 = web.DataReader("^N225", "yahoo").reset_index()[['Date', 'Adj Close']].rename(columns={'Date': '年月日', 'Adj Close':'N225'})
# AXJO = web.DataReader("^AXJO", "yahoo").reset_index()[['Date', 'Adj Close']].rename(columns={'Date': '年月日', 'Adj Close':'AXJO'})
# GSPC = web.DataReader("^GSPC", "yahoo").reset_index()[['Date', 'Adj Close']].rename(columns={'Date': '年月日', 'Adj Close':'GSPC'})
# IXIC = web.DataReader("^IXIC", "yahoo").reset_index()[['Date', 'Adj Close']].rename(columns={'Date': '年月日', 'Adj Close':'IXIC'})
# GDAXI = web.DataReader("^GDAXI", "yahoo").reset_index()[['Date', 'Adj Close']].rename(columns={'Date': '年月日', 'Adj Close':'GDAXI'})
# # FTSE = web.DataReader("^FTSE", "yahoo").reset_index()[['Date', 'Adj Close']].rename(columns={'Date': '年月日', 'Adj Close':'FTSE'})
# STOXX50E = web.DataReader("^STOXX50E", "yahoo").reset_index()[['Date', 'Adj Close']].rename(columns={'Date': '年月日', 'Adj Close':'STOXX50E'})
# l = [TWII, SSE, HSI, N225, AXJO, GSDIC, IXIC, GDAXI, STOXX50E]
# index = cytoolz.reduce(mymerge, l).sort_values(['年月日'])
# index.年月日=pd.to_datetime(index.年月日).apply(lambda x: x.date())
# print('index')

forweb = m[col]
#---- bic ----
conn_lite = conn_local_lite('bic.sqlite3')
cur_lite = conn_lite.cursor()
sql = "SELECT * FROM '{}'"
bic = pd.read_sql_query(sql.format('景氣指標及燈號-指標構成項目'), conn_lite)
del bic['年月']
m['年月日'] = m['年月日'].astype(str)
m['年'], m['月'] = m['年月日'].str.split('-').str[0].astype(int), m['年月日'].str.split('-').str[1].astype(int)
# m.dtypes
m = mymerge(m, bic)
del m['年'], m['月'], bic['年'], bic['月']
m.年月日 = pd.to_datetime(m.年月日, format='%Y/%m/%d').apply(lambda x: x.date())

forweb = m[col+list(bic)]
forweb['d'] = forweb['調整收盤價'] - forweb['收盤價']
forweb['調整開盤價'] = forweb['開盤價'] + forweb.d
forweb['調整收盤價'] = forweb['收盤價'] + forweb.d
forweb['調整最高價'] = forweb['最高價'] + forweb.d
forweb['調整最低價'] = forweb['最低價'] + forweb.d
forweb['earning'] = forweb['收盤價']/forweb['本益比']
forweb['lnmo'] = log(forweb['調整收盤價']/forweb['調整收盤價'].shift(120))

# return
@timeSpan
def Return(*period):
    for days in period:
        n = 240/days
        forweb['r' + str(days)] = (forweb['調整收盤價'].shift(-days)/forweb['調整收盤價'])**n-1
Return(5, 10, 20, 40, 60, 120)

# log return
@timeSpan
def lnReturn(*period):
    for days in period:
        n = 240/days
        forweb['lnr' + str(days)] = log(forweb['調整收盤價'].shift(-days)/forweb['調整收盤價'])*n
lnReturn(5, 10, 20, 40, 60, 120)

# return standard deviation
@timeSpan
def ReturnStd(*period):
    for days in period:
        name = 'r' + str(days)
        forweb[name + 'Std'] = (forweb[name]-forweb[name].mean())/forweb[name].std()
ReturnStd(5, 10, 20, 40, 60, 120)

# rsi
forweb['ch'] = forweb['調整收盤價'].diff()
forweb['ch_u'], forweb['ch_d'] = forweb['ch'], forweb['ch']
forweb.ix[forweb.ch_u < 0, 'ch_u'], forweb.ix[forweb.ch_d > 0, 'ch_d'] = 0, 0
forweb['ch_d'] = forweb['ch_d'].abs()
forweb['rsi'] = forweb.ch_u.ewm(alpha=1/14).mean()/(forweb.ch_u.ewm(alpha=1/14).mean()+forweb.ch_d.ewm(alpha=1/14).mean())*100 #與r和凱基同,ema的公式與一般的ema不同。公式見http://www.fmlabs.com/reference/default.htm?url=RSI.htm
forweb = forweb.drop(['ch', 'ch_u', 'ch_d'], axis=1)

# ma
@timeSpan
def ma(*period):
    for n in period:
        forweb['MA'+str(n)] = forweb['收盤價'].rolling(window=n).mean()
@timeSpan
def ma_adj(*period):
    for n in period:
        forweb['MA'+str(n)+'.adj'] = forweb['調整收盤價'].rolling(window=n).mean()
ma(5, 10, 20, 60, 120)
ma_adj(5, 10, 20, 60, 120)

# DI
forweb['DI'] = (forweb['最高價']+forweb['最低價']+2*forweb['收盤價'])/4
forweb['DI.adj'] = (forweb['調整最高價']+forweb['調整最低價']+2*forweb['調整收盤價'])/4

# macd
forweb['max9'] = forweb['最高價'].rolling(window=9).max()
forweb['min9'] = forweb['最低價'].rolling(window=9).min()
forweb['EMA12'] = forweb.DI.ewm(alpha=2/13).mean()
forweb['EMA26'] = forweb.DI.ewm(alpha=2/27).mean()
forweb['DIF'] = forweb['EMA12']-forweb['EMA26']
forweb['MACD'] = forweb.DIF.ewm(alpha=0.2).mean()
forweb['MACD1'] = (forweb['EMA12']-forweb['EMA26'])/forweb['EMA26']*100
forweb['OSC'] = forweb.DIF - forweb.MACD

# bband
forweb['std5'] = forweb['DI'].rolling(window=5).std()
forweb['std10'] = forweb['DI'].rolling(window=11).std()
forweb['std20'] = forweb['DI'].rolling(window=20).std()
forweb['mavg'] = forweb['DI'].rolling(window=20).mean()
forweb['up'] = forweb.mavg + forweb['std20']*2
forweb['dn'] = forweb.mavg - forweb['std20']*2
forweb['bband'] = (forweb['收盤價']-forweb.mavg)/forweb['std20']

# bband adj
forweb['std5.adj'] = forweb['DI.adj'].rolling(window=5).std()
forweb['std10.adj'] = forweb['DI.adj'].rolling(window=11).std()
forweb['std20.adj'] = forweb['DI.adj'].rolling(window=20).std()
forweb['mavg.adj'] = forweb['DI.adj'].rolling(window=20).mean()
forweb['up.adj'] = forweb['mavg.adj'] + forweb['std20.adj']*2
forweb['dn.adj'] = forweb['mavg.adj'] - forweb['std20.adj']*2
forweb['bband.adj'] = (forweb['調整收盤價']-forweb['mavg.adj'])/forweb['std20.adj']

# kd
forweb['rsv'] = (forweb['收盤價']-forweb.min9)/(forweb.max9-forweb.min9)
forweb['k'] = forweb.rsv.ewm(alpha=1/3).mean()
forweb['d'] = forweb.k.ewm(alpha=1/3).mean()

# others
forweb['high-low'] = (forweb['最高價']-forweb['最低價'])/forweb['收盤價']
forweb['pch'] = (forweb['收盤價']-forweb['收盤價'].shift())/forweb['收盤價'].shift()
forweb['pctB'] = (forweb.DI-forweb.dn)/(forweb.up-forweb.dn)
forweb['close-up'] = (forweb['收盤價']-forweb.up)/(forweb.DI.rolling(window=20).std()*2)
forweb['close-dn'] = (forweb['收盤價']-forweb.dn)/(forweb.DI.rolling(window=20).std()*2)

forweb['pctB.adj'] = (forweb['DI.adj']-forweb['dn.adj'])/(forweb['up.adj']-forweb['dn.adj'])
forweb['close-up.adj'] = (forweb['調整收盤價']-forweb['up.adj'])/(forweb['DI.adj'].rolling(window=20).std()*2)
forweb['close-dn.adj'] = (forweb['調整收盤價']-forweb['dn.adj'])/(forweb['DI.adj'].rolling(window=20).std()*2)

timeDelta('before trend')

# def pch_column(df, column):
#     df1 = deepcopy(df)
#     df1['pch_{}'.format(column)] = df1[column].pct_change()
#     return df1
#
# def pch_columns(df, columns):
#     pch_df = functools.partial(pch_column, df)
#     return cytoolz.reduce(mymerge, map(pch_df, columns))
@timeSpan
def pch(df, columns):
    df1 = deepcopy(df)
    for col in columns:
        df1['pch_{}'.format(col)] = df1[col].pct_change()
    return df1
@timeSpan
def trend(df, columns):
    df1 = deepcopy(df)
    for col in columns:
        df1['trend_{}'.format(col)] = sign(df1['pch_{}'.format(col)])
        i = df1[df1['trend_{}'.format(col)] == 0].index
        while i.tolist() != []:
            df1.ix[i, 'trend_{}'.format(col)] = df1.ix[i - 1, 'trend_{}'.format(col)].tolist()
            i = df1[df1['trend_{}'.format(col)] == 0].index
    return df1
@timeSpan
def reversion(df, columns):
    df1 = deepcopy(df)
    for col in columns:
        # init reversion
        df1['reversion_{}'.format(col)] = df1['trend_{}'.format(col)] - df1['trend_{}'.format(col)]

        # trend reverse to positive
        i = df1[df1['trend_{}'.format(col)] == 1].index
        a = np.array(i)
        l = (a[1:] - a[:-1]).tolist()
        i = np.array([i for i, j in enumerate(l) if j != 1]) + 1
        df1.ix[a[i], 'reversion_{}'.format(col)] = 1

        # trend reverse to negtive
        i = df1[df1['trend_{}'.format(col)] == -1].index
        a = np.array(i)
        l = (a[1:] - a[:-1]).tolist()
        i = np.array([i for i, j in enumerate(l) if j != 1]) + 1
        df1.ix[a[i], 'reversion_{}'.format(col)] = -1

        # first reversion
        i = df1.ix[df1['trend_{}'.format(col)] == 1].index[0]
        if df1.ix[i, 'trend_{}'.format(col)]>df1.ix[i-1, 'trend_{}'.format(col)] and df1.ix[i, 'trend_{}'.format(col)] !=0:
            df1.ix[i, 'reversion_{}'.format(col)] = 1
        i = df1.ix[df1['trend_{}'.format(col)] == -1].index[0]
        if df1.ix[i, 'trend_{}'.format(col)]<df1.ix[i-1, 'trend_{}'.format(col)] and df1.ix[i, 'trend_{}'.format(col)] !=0:
            df1.ix[i, 'reversion_{}'.format(col)] = -1
        # print(df1[['pch_{}'.format(col), 'trend_{}'.format(col), 'reversion_{}'.format(col)]].head(100))
    return df1

forweb = pch(forweb, ['收盤價', 'MA5', 'MA10', 'MA20', 'MA60', 'MA120'])
forweb = trend(forweb, ['收盤價', 'MA5', 'MA10', 'MA20', 'MA60', 'MA120'])
forweb = reversion(forweb, ['收盤價', 'MA5', 'MA10', 'MA20', 'MA60', 'MA120'])
@timeSpan
def local_min_or_max(df, columns):
    df1 = deepcopy(df)
    for col in columns:
        df1['local_min(max)_{}'.format(col)] = df1['reversion_{}'.format(col)] - df1['reversion_{}'.format(col)]
        i = df1.ix[df1['reversion_{}'.format(col)] == 1].index
        a = np.array(i)
        df1.ix[a - 1, 'local_min(max)_{}'.format(col)] = df1.ix[a, 'reversion_{}'.format(col)].tolist()
        i = df1.ix[df1['reversion_{}'.format(col)] == -1].index
        a = np.array(i)
        df1.ix[a - 1, 'local_min(max)_{}'.format(col)] = df1.ix[a, 'reversion_{}'.format(col)].tolist()
    return df1

forweb = local_min_or_max(forweb, ['收盤價', 'MA5', 'MA10', 'MA20', 'MA60', 'MA120'])
@timeSpan
def new_high_or_low(df, columns):
    df1 = deepcopy(df)
    for col in columns:
        df1['new_high(low)_{}'.format(col)] = df1['local_min(max)_{}'.format(col)] - df1['local_min(max)_{}'.format(col)]
        i = df1.ix[df1['local_min(max)_{}'.format(col)] == 1, 'local_min(max)_{}'.format(col)].index.tolist()
        a = np.array(i)
        l = (df1['{}'.format(col)][a] - df1['{}'.format(col)][a].shift()).tolist()
        i = np.array([i for i, j in enumerate(l) if j > 0])
        df1.ix[a[i], 'new_high(low)_{}'.format(col)] = 1
        i = df1.ix[df1['local_min(max)_{}'.format(col)] == -1, 'local_min(max)_{}'.format(col)].index.tolist()
        a = np.array(i)
        l = (df1['{}'.format(col)][a] - df1['{}'.format(col)][a].shift()).tolist()
        i = np.array([i for i, j in enumerate(l) if j < 0])
        df1.ix[a[i], 'new_high(low)_{}'.format(col)] = -1
    return df1

forweb = new_high_or_low(forweb, ['收盤價', 'MA5', 'MA10', 'MA20', 'MA60', 'MA120'])

list(forweb)

forweb[['local_min(max)_MA5', 'new_high(low)_MA5']].head(100)

forweb['span'] = abs(forweb['調整收盤價']-forweb.調整開盤價)/forweb['調整收盤價']
forweb['span_high-low'] = abs(forweb['調整最高價']-forweb['調整最低價'])/forweb['調整收盤價']
forweb['upperShadow'] = (forweb['調整最高價'] - forweb[['調整開盤價', '調整收盤價']].max(axis=1))/forweb['調整收盤價']
forweb['lowerShadow'] = (forweb[['調整開盤價', '調整收盤價']].min(axis=1) - forweb['調整最低價'])/forweb['調整收盤價']
forweb['upperShadow/span'] =forweb['upperShadow']/(forweb['span']+0.1**10*forweb['調整收盤價'])
forweb['lowerShadow/span'] =forweb['lowerShadow']/(forweb['span']+0.1**10*forweb['調整收盤價'])
# forweb['span/upperShadow'] =forweb['span']/forweb['upperShadow']
# forweb['span/lowerShadow'] =forweb['span']/forweb['lowerShadow']
forweb['span/(high-low)'] =forweb['span']/forweb['span_high-low']
del forweb['d']
forweb['high-low_1ag1'] = forweb['high-low'].shift()
forweb['high-low_lag2'] = forweb['high-low'].shift(2)
forweb['upperShadow_lag1'] = forweb['upperShadow'].shift()
forweb['lowerShadow_lag1'] = forweb['lowerShadow'].shift()
forweb['upperShadow/span_lag1'] = forweb['upperShadow/span'].shift()
forweb['lowerShadow/span_lag1'] = forweb['lowerShadow/span'].shift()
# forweb['span/upperShadow_lag1'] = forweb['span/upperShadow'].shift()
# forweb['span/lowerShadow_lag1'] = forweb['span/lowerShadow'].shift()
forweb['spandiff'] = forweb.span.diff()
forweb['spanudiff'] = forweb[['調整開盤價', '調整收盤價']].max(axis=1).diff()
forweb['spanldiff'] = forweb[['調整開盤價', '調整收盤價']].min(axis=1).diff()
forweb['span/(high-low)_lag1'] = forweb['span/(high-low)'].shift()

timeDelta('before OSCsign')

forweb['OSCsign'] = sign(forweb.OSC)
forweb['gr'] = 0

OSCsign = forweb['OSCsign'].tolist()
gr = forweb['gr'].tolist()
g = 0
for i in range(len(OSCsign)-1):
    if OSCsign[i]*OSCsign[i+1] < 0:
        g+=1
        gr[i+1] = g
    else:
        gr[i+1] = g

forweb['OSCsign'], forweb['gr'] = OSCsign, gr
del g, OSCsign, gr
@timeSpan
def minORmax(df):
    df1 = deepcopy(df)
    if df1.max()>0:
        return df1.max()
    if df1.min()<0:
        return df1.min()
    else:
        return df1

grouped = forweb.groupby('gr')
l = grouped['OSC'].apply(minORmax).tolist()

d = {}
for i, v in enumerate(l):
    d[i+2] = v
d[0], d[1] = np.nan, np.nan
forweb[['gr1']] = forweb[['gr']].applymap(lambda x:d[x])

forweb['change'] = 0
@timeSpan
def OSCbreakpoint(df):
    df1 = deepcopy(df)
    df1=df1.reset_index(drop=True)  # without this df1.ix[0,'gr1'] is only defined in first group
    if df1['OSC'].max()>0:
        for i in range(len(df1['gr1'])):
            # print(i, len(df1['gr1']))
            # print(i, df1.ix[i,'OSC'], df1.ix[i,'gr1'])
            if df1.ix[i,'OSC']>df1.ix[i,'gr1']:
                # print(i, 'yes')
                df1.ix[i, 'change'] = 1
                break
        return df1
    if df1['OSC'].min()<0:
        for i in range(len(df1['gr1'])):
            # print(i, len(df1['gr1']))
            # print(i, df1.ix[i,'OSC'], df1.ix[i,'gr1'])
            if df1.ix[i,'OSC']<df1.ix[i,'gr1']:
                # print(i, 'yes')
                df1.ix[i, 'change'] = -1
                break
        return df1
    else:
        return df1

forweb = grouped.apply(OSCbreakpoint).reset_index(drop=True)
del forweb['OSCsign'], forweb['gr'], forweb['gr1']

timeDelta('forweb')

tablename = 'demo'
# forweb = mymerge(forweb, index).sort_values(['年月日'])
forweb['漲跌(+/-)'] = forweb['漲跌(+/-)'].replace('＋', 1).replace('－', -1).replace('X', 0).replace(' ', None).astype(float)
forweb['外資鉅額交易'] = forweb['外資鉅額交易'].replace('yes', 1).replace('no', 0).astype(float)
forweb['投信鉅額交易'] = forweb['投信鉅額交易'].replace('yes', 1).replace('no', 0).astype(float)
forweb.年月日 = forweb.年月日.astype(str)
forweb.證券代號 = forweb.證券代號.astype(str)
forweb = forweb.drop_duplicates(['年月日', '證券代號'])
# list(forweb)
conn_lite = conn_local_lite('mysum.sqlite3')
cur_lite = conn_lite.cursor()

sql = 'ALTER TABLE `{}` RENAME TO `{}0`'.format(tablename, tablename)
cur_lite.execute(sql)
sql = 'create table `{}` (`{}`, PRIMARY KEY ({}))'.format(tablename, '`,`'.join(list(forweb)), '`年月日`, `證券代號`')
cur_lite.execute(sql)
sql = 'insert into `{}`(`{}`) values({})'.format(tablename, '`,`'.join(list(forweb)), ','.join('?'*len(list(forweb))))
cur_lite.executemany(sql, forweb.values.tolist())
conn_lite.commit()
sql = "drop table `{}0`".format(tablename)
cur_lite.execute(sql)

# sql = 'DROP TABLE forweb'
# cur_lite.execute(sql)
# forweb.to_sql('forweb', conn_lite, index=False)
# list(forweb)
# forweb.to_sql('forweb1', conn_local_lite('C:/Users/ak66h_000/OneDrive/webscrap/djangogirls/mysite/db.sqlite3'))
# forweb.to_sql('forweb', conn_local_lite('C:/Users/ak66h_000/OneDrive/testpydev/src/db.sqlite3'), index=False)
# forweb.to_csv('C:/Users/ak66h_000/Dropbox/forspark.csv', index=False)
# forweb.to_json('C:/Users/ak66h_000/Dropbox/forspark.json',force_ascii=False)
forweb = forweb.reset_index(drop=True)
timeDelta('finish')