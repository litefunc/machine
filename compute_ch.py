from sql.pg import select, insert, delete
from common.connection import conn_local_pg
from common.env import PG_PWD, PG_PORT, PG_USER
import syspath
import pandas as pd
import numpy as np
import cytoolz.curried
import datetime as dt
import os
import sys
if os.getenv('MY_PYTHON_PKG') not in sys.path:
    sys.path.append(os.getenv('MY_PYTHON_PKG'))


def timeSpan(func):
    def wrapper(*args, **kw):
        print('call %s():' % func.__name__)
        start = dt.datetime.now()
        x = func(*args, **kw)
        end = dt.datetime.now()
        print('Complete in {} second(s)'.format(end-start))
        return x
    return wrapper

#df2 = select.sw(["年月日", "證券代號", "開盤價", "最高價", "最低價", "收盤價"], '每日收盤行情(全部(不含權證、牛熊證))', {'證券代號':'5522'}).df(conn_local_pg('tse'))
#df3 = select.sw(["年月日", "證券代號", "權值+息值"], '除權息計算結果表', {'證券代號':'5522'}).df(conn_local_pg('tse'))


def mymerge(x, y):
    m = pd.merge(x, y, on=[col for col in list(x)
                           if col in list(y)], how='outer')
    return m

#df4 = mymerge(df2, df3)


sql = f'''select a."年月日", a."證券代號", a."開盤價", a."最高價", a."最低價", a."收盤價", b."權值+息值" from "每日收盤行情(全部(不含權證、牛熊證))" as a left join "除權息計算結果表" as b on
 a."年月日"=b."年月日" and  a."證券代號"=b."證券代號"
'''
df = pd.read_sql_query(sql, conn_local_pg('tse'))
df['adj'] = df['權值+息值'].fillna(0)

keys = ['證券代號']
#g = df.groupby(keys)
df['adjcum'] = df.groupby(keys)['adj'].apply(lambda x: x.cumsum())
df['收盤價:調整'] = df['收盤價'] + df['adjcum']
df['開盤價:調整'] = df['開盤價'] + df['adjcum']
df['最高價:調整'] = df['最高價'] + df['adjcum']
df['最低價:調整'] = df['最低價'] + df['adjcum']
df = df.drop(['adj', 'adjcum', '權值+息值'], axis=1)

# price
df['price'] = (df['最高價']+df['最低價']+2*df['收盤價'])/4
df['price:調整'] = (df['最高價:調整']+df['最低價:調整']+2*df['收盤價:調整'])/4

# standard deviation


def stdev(df, keys, cols, periods):
    for col in cols:
        for p in periods:
            df[f'{col}:標準差{p}'] = df.groupby(keys)[col].apply(
                lambda df: df.rolling(window=p).std())
    return df


df = stdev(df, keys, ['price', 'price:調整'], [20])

# macd, osc
df['EMA12'] = df.groupby(keys)['price'].apply(
    lambda x: x.ewm(alpha=2/13).mean())
df['EMA26'] = df.groupby(keys)['price'].apply(
    lambda x: x.ewm(alpha=2/27).mean())
df['DIF'] = df['EMA12']-df['EMA26']
df['MACD'] = df.groupby(keys)['DIF'].apply(lambda x: x.ewm(alpha=0.2).mean())
df['MACD1'] = (df['EMA12']-df['EMA26'])/df['EMA26']*100
df['OSC'] = df['DIF'] - df['MACD']
df = df.drop(['EMA12', 'EMA26'], axis=1)


# b band
df['Avg_Band'] = df.groupby(keys)['price'].apply(
    lambda df: df.rolling(window=20).mean())
df['Upper_Band'] = df.groupby(keys).apply(
    lambda df: df['Avg_Band'] + df['price:標準差20']*2).reset_index(drop=True)
df['Lower_Band'] = df.groupby(keys).apply(
    lambda df: df['Avg_Band'] - df['price:標準差20']*2).reset_index(drop=True)
df['(price-Avg_Band)/price:標準差20'] = df.groupby(keys).apply(lambda df: (df['price'] -
                                                                        df['Avg_Band'])/df['price:標準差20']).reset_index(drop=True)

# b band adj
df['Avg_Band:調整'] = df.groupby(keys)['price:調整'].apply(
    lambda df: df.rolling(window=20).mean())
df['Upper_Band:調整'] = df.groupby(keys).apply(
    lambda df: df['Avg_Band:調整'] + df['price:調整:標準差20']*2).reset_index(drop=True)
df['Lower_Band:調整'] = df.groupby(keys).apply(
    lambda df: df['Avg_Band:調整'] - df['price:調整:標準差20']*2).reset_index(drop=True)
df['(price:調整-Avg_Band:調整)/price:調整:標準差20'] = df.groupby(keys).apply(lambda df: (
    df['price:調整']-df['Avg_Band:調整'])/df['price:調整:標準差20']).reset_index(drop=True)


# return
@timeSpan
def Return(df, keys, cols, periods):
    for days in periods:
        n = 240/days
        __cols = [f'報酬率{days}' for col in cols]
        df[__cols] = df.groupby(keys)[cols].apply(
            lambda df: (df.shift(-days)/df)**n-1)
    return df


df = Return(df, keys, ['收盤價'], [5, 10, 20, 40, 60, 120])


@timeSpan
def Return_adj(df, keys, cols, periods):
    for days in periods:
        n = 240/days
        __cols = [f'報酬率{days}:調整' for col in cols]
        df[__cols] = df.groupby(keys)[cols].apply(
            lambda df: (df.shift(-days)/df)**n-1)
    return df


df = Return_adj(df, keys, ['收盤價:調整'], [5, 10, 20, 40, 60, 120])

# log return
@timeSpan
def lnReturn(df, keys, cols, periods):
    for days in periods:
        n = 240/days
        __cols = [f'ln報酬率{days}' for col in cols]
        df[__cols] = df.groupby(keys)[cols].apply(
            lambda df: np.log(df.shift(-days)/df)*n)
    return df


df = lnReturn(df, keys, ['收盤價'], [5, 10, 20, 40, 60, 120])


@timeSpan
def lnReturn_adj(df, keys, cols, periods):
    for days in periods:
        n = 240/days
        __cols = [f'ln報酬率{days}:調整' for col in cols]
        df[__cols] = df.groupby(keys)[cols].apply(
            lambda df: np.log(df.shift(-days)/df)*n)
    return df


df = lnReturn_adj(df, keys, ['收盤價:調整'], [5, 10, 20, 40, 60, 120])


@timeSpan
def Normalize(df, keys, cols):
    __cols = [f'{col}:nmz' for col in cols]
    df[__cols] = df.groupby(keys)[cols].apply(
        lambda df: (df-df.mean())/df.std())
    return df


#df = Normalize(df, keys, ['r5:收盤價', 'r10:收盤價', 'r20:收盤價', 'r40:收盤價', 'r60:收盤價', 'r120:收盤價'])
#df = Normalize(df, keys, ['r5:收盤價:調整', 'r10:收盤價:調整', 'r20:收盤價:調整', 'r40:收盤價:調整', 'r60:收盤價:調整', 'r120:收盤價:調整'])

# ma
@timeSpan
def ma(df, keys, cols, period):
    for n in period:
        __cols = [f'MA{n}'for col in cols]
        df[__cols] = df.groupby(keys)[cols].apply(
            lambda df: df.rolling(window=n).mean())
    return df


df = ma(df, keys, ['收盤價'], [5, 10, 20, 40, 60, 120])


@timeSpan
def ma_adj(df, keys, cols, period):
    for n in period:
        __cols = [f'MA{n}:調整'for col in cols]
        df[__cols] = df.groupby(keys)[cols].apply(
            lambda df: df.rolling(window=n).mean())
    return df


df = ma_adj(df, keys, ['收盤價:調整'], [5, 10, 20, 40, 60, 120])

# rsi
df['ch'] = df.groupby(keys)['收盤價'].apply(lambda x: x.diff())
df['ch_u'], df['ch_d'] = df['ch'], df['ch']
df.loc[df['ch_u'] < 0, 'ch_u'] = 0
df.loc[df['ch_d'] > 0, 'ch_d'] = 0
df['ch_d'] = df['ch_d'].abs()
#df['rsi'] = g['ch_u'].apply(lambda x: x.ewm(alpha=1/14).mean()) / (g['ch_u'].apply(lambda x: x.ewm(alpha=1/14).mean()) + g['ch_d'].apply(lambda x: x.ewm(alpha=1/14).mean()))*100
df['RSI'] = df.groupby(keys).apply(lambda df: df['ch_u'].ewm(alpha=1/14).mean() / (
    df['ch_u'].ewm(alpha=1/14).mean() + df['ch_d'].ewm(alpha=1/14).mean())*100).reset_index(drop=True)
df = df.drop(['ch', 'ch_u', 'ch_d'], axis=1)

# kdj
df['max9'] = df.groupby(keys)['最高價'].apply(
    lambda df: df.rolling(window=9).max())
df['min9'] = df.groupby(keys)['最低價'].apply(
    lambda df: df.rolling(window=9).min())
df['rsv'] = df.groupby(keys).apply(lambda df: (
    df['收盤價']-df['min9'])/(df['max9']-df['min9'])).reset_index(drop=True)
df['K'] = df.groupby(keys)['rsv'].apply(lambda df: df.ewm(alpha=1/3).mean())
df['D'] = df.groupby(keys)['K'].apply(lambda df: df.ewm(alpha=1/3).mean())
df['J'] = 3*df['D']-2*df['K']
df = df.drop(['max9', 'min9', 'rsv'], axis=1)

df = df.drop(['開盤價', '最高價', '最低價', '收盤價',
              'price:標準差20', 'price:調整:標準差20'], axis=1)
list(df)

date_cols = ['年月日']
varchar_cols = ['證券代號']
real_cols = [col for col in list(df) if col not in (date_cols + varchar_cols)]
cols = date_cols + varchar_cols + real_cols

# replace big number with np.inf, otherwise postgresql will show `value out of range: overflow`
for col in real_cols:
    df.loc[df[col] > 10**15, col] = np.inf

# df.dtypes

tse = conn_local_pg('tse')
table = 'compute'

# create table
#p_keys = ', '.join(['年月日', '證券代號'])
#dtypes = {'date':date_cols, 'varchar(14)':varchar_cols, 'real':real_cols}
#
#__columns = []
# for t in dtypes.keys():
#    __columns = __columns + [f'"{col}" {t}' for col in dtypes[t]]
#
#columns = ', '.join(__columns)
#sql = f'create table "{table}" ({columns}, PRIMARY KEY({p_keys}))'
#cur = tse.cursor()
# cur.execute(sql)

# g.get_group('5522')

delete.Delete('compute').run(tse)

g = df.round(4).groupby(keys)
start = dt.datetime.now()
for group in g.groups:
    print(group)
    insert.Insert(table, cols).run(tse, g.get_group(group))

print('Complete in {} second(s)'.format(dt.datetime.now()-start))

tse.commit()
