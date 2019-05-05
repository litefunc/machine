import dftosql
import astype as ast
import sqlCommand as sqlc
from common.connection import conn_local_pg
import syspath
import psycopg2 as pg
import time
from datetime import datetime, timedelta
from copy import deepcopy
import cytoolz.curried
import pandas as pd
import numpy as np
import os
import sys
from sql.pg import select
from pymongo import MongoClient


if os.getenv('MY_PYTHON_PKG') not in sys.path:
    sys.path.append(os.getenv('MY_PYTHON_PKG'))


def mymerge(x, y):
    m = pd.merge(x, y, on=[col for col in list(x)
                           if col in list(y)], how='outer')
    return m


def s_by(conn: pg.extensions.connection, table: str, col: str, id: str) -> pd.DataFrame:
    companyId = "'{}'".format(id)
    sql = 'SELECT * FROM "{}" WHERE "{}"={}'.format(table, col, companyId)
    print(sql)
    return pd.read_sql_query(sql, conn)


def s_by_companyid(conn: pg.extensions.connection, table: str, id: str) -> pd.DataFrame:
    companyId = "'{}'".format(id)
    sql = 'SELECT * FROM "{}" WHERE 公司代號={}'.format(table, companyId)
    print(sql)
    return pd.read_sql_query(sql, conn)


def s_by_id(conn: pg.extensions.connection, table: str, id: str) -> pd.DataFrame:
    companyId = "'{}'".format(id)
    sql = 'SELECT * FROM "{}" WHERE 證券代號={}'.format(table, companyId)
    print(sql)
    return pd.read_sql_query(sql, conn)


# --- summary ---

def account(conn, id):
    df = s_by_companyid(conn, '會計師查核報告', id).rename(columns={'公司代號': '證券代號', '核閱或查核日期': '年月日'}).sort_values(
        ['年', '季', '證券代號']).drop(['公司簡稱', '簽證會計師事務所名稱', '簽證會計師', '簽證會計師.1', '核閱或查核報告類型'], axis=1)
    df[['年', '季']] = df[['年', '季']].astype(str)
    return df


def finance(conn, id):
    df = s_by_companyid(conn, '財務分析', id).drop(['公司簡稱'], axis=1)
    #del fin['公司簡稱']
    return df


# --- mops---

def de_accumulation(df, columns):
    df0 = df[columns]
    df1 = df[list(filter(lambda x: x not in columns, list(df)))]
    a0 = np.array(df0)
    a1 = np.array(df1)
    # season 4 - season 3, season 3 - season 2, season 2 - season 1, season 1 remains intact instead of nan
    # make sure season is accend
    v = np.vstack((a1[0], a1[1:] - a1[0:len(df) - 1]))
    h = np.hstack((a0, v))
    return pd.DataFrame(h, columns=list(df0) + list(df1))


def income(conn, id):
    df = s_by_companyid(conn, 'ifrs前後-綜合損益表', id)
    df.dtypes
    floatColumns = list(
        filter(lambda x: x not in ['年', '季', '公司代號', '公司名稱'], list(df)))
    df[floatColumns] = df[floatColumns].astype(float)
    # df = df.groupby(['公司代號', '年']).apply(change).reset_index(drop=True)  #'季' must be string
    df = df.groupby(['公司代號', '年']).apply(de_accumulation, [
        '年', '季', '公司代號', '公司名稱']).reset_index(drop=True)  # '季' must be string
    df['grow_s'] = df['本期綜合損益總額'].pct_change(1)
    df['grow_hy'] = df['本期綜合損益總額'].rolling(window=2).sum().pct_change(2)
    # df[col1] = df[col1].rolling(window=4).sum()
    df[floatColumns] = df[floatColumns].rolling(window=4).sum()
    df['grow_y'] = df['本期綜合損益總額'].pct_change(4)
    df['grow'] = df['本期綜合損益總額'].pct_change(1)
    # df['grow.ma'] = df['grow'].rolling(window=24).mean()*4
    df['本期綜合損益總額.wma'] = df['本期綜合損益總額'].ewm(com=19).mean() * 4
    df['本期綜合損益總額.ma'] = df['本期綜合損益總額'].rolling(window=12).mean() * 4
    df['毛利率'] = df['營業毛利（毛損）']/df['營業收入']
    df['營業利益率'] = df['營業利益（損失）']/df['營業收入']
    df['綜合稅後純益率'] = df['綜合損益總額歸屬於母公司業主']/df['營業收入']

    return df


def balance(conn, id):
    sql = "SELECT * FROM '{}' WHERE 公司代號 LIKE {}"
    df = s_by_companyid(conn, 'ifrs前後-資產負債表-一般業', id).drop(
        ['公司名稱', 'Unnamed: 21', '待註銷股本股數（單位：股）', 'Unnamed: 22'], axis=1)
    df[['年', '季']] = df[['年', '季']].astype(str)
    return df


def report(inc, bal):
    df = mymerge(inc, bal)
    df['流動比率'] = df['流動資產'] / df['流動負債']
    df['負債佔資產比率'] = df['負債總額'] / df['資產總額']
    df['權益報酬率'] = df['綜合損益總額歸屬於母公司業主'] * 2 / (df['權益總額'] + df['權益總額'].shift())
    df['profitbility'] = df['綜合損益總額歸屬於母公司業主'] / (df['權益總額'].shift(4))
    df['investment'] = df['權益總額'].pct_change(4)
    df = df.rename(columns={'公司代號': '證券代號'})

    return df


# --- tse ---

def tsedata(conn, id):
    close = s_by_id(conn, '每日收盤行情(全部(不含權證、牛熊證))', id)
    value = s_by_id(conn, '個股日本益比、殖利率及股價淨值比', id).drop(['證券名稱'], 1)
    margin = s_by_id(conn, '當日融券賣出與借券賣出成交量值(元)', id)
    ins = s_by_id(conn, '三大法人買賣超日報', id)
    fore = s_by_id(conn, '外資及陸資買賣超彙總表 (股)', id).drop(['證券名稱'], 1).rename(columns={
        '買進股數': '外資買進股數', '賣出股數': '外資賣出股數', '買賣超股數': '外資買賣超股數', '鉅額交易': '外資鉅額交易'})
    trust = s_by_id(conn, '投信買賣超彙總表 (股)', id).drop(['證券名稱'], 1).rename(columns={
        '買進股數': '投信買進股數', '賣出股數': '投信賣出股數', '買賣超股數': '投信買賣超股數', '鉅額交易': '投信鉅額交易'})
    index = pd.read_sql_query('SELECT * FROM "{}" '.format('index'), conn)
    xdr = s_by_id(conn, '除權息計算結果表', id)
    df = cytoolz.reduce(mymerge, [close, value, fore, trust, index, xdr])
    return df


# --- merge ---

# if value is null, set it to previous value
def fill(s):
    a = np.array(0)
    notnull = s[~pd.isnull(s)].index
    a = np.append(a, notnull)
    a = np.append(a, len(s))  # [0, *notnull, len(s)]
    len_notnull = a[1:] - a[:len(a) - 1]
    l = []
    for i in range(len(len_notnull)):
        l = l + np.repeat(s[a[i]], len_notnull[i]).tolist()
    return pd.Series(l, name=s.name)


def merge(tse, ac, report):
    df = cytoolz.reduce(mymerge, [tse, ac, report])
    # should convert to datetime before sort, or the result is wrong
    df.年月日 = pd.to_datetime(
        df.年月日, format='%Y/%m/%d').apply(lambda x: x.date())
    df = df.sort_values(['年月日', '證券代號']).reset_index(
        drop=True)  # reset_index make the index ascending

    df[list(ac)+list(report)] = df[list(ac)+list(report)].apply(fill)
    df['time'] = df.index.tolist()
    del df['財報年/季']

    floatColumns = [col for col in list(df) if col not in [
        '年月日', '證券代號', '年', '季', '證券名稱', '公司名稱']]
    df[floatColumns] = df[floatColumns].astype(float)

    df = df[['年月日', '證券代號', '年', '季'] +
            [x for x in list(df) if x not in ['年月日', '證券代號', '年', '季']]]

    df['adj'] = df['權值+息值'].replace(np.nan, 0)

    df['adjcum'] = df['adj'].cumsum()
    df['調整收盤價'] = df['收盤價']+df['adjcum']
    df['調整開盤價'] = df['開盤價']+df['adjcum']
    df['調整最高價'] = df['最高價']+df['adjcum']
    df['調整最低價'] = df['最低價']+df['adjcum']

    df = df.drop(['adj', 'adjcum'], axis=1)
    df['earning'] = (df['收盤價']/df['本益比']).replace(np.inf,
                                                  0)  # without this earning can be inf
    df['lnmo'] = np.log(df['調整收盤價']/df['調整收盤價'].shift(120))

    df = df.dropna(axis=1, how='all')

    return df


# ---- bic ----

def businese(conn):
    df = pd.read_sql_query(
        'SELECT * FROM "{}"'.format('景氣指標及燈號-指標構成項目'), conn).drop(['年月'], 1)
    df['年'] = df['年'].astype(int)
    df['月'] = df['月'].astype(int)
    return df


def mergebic(m, bic):
    m['年月日'] = m['年月日'].astype(str)
    m['年'], m['月'] = m['年月日'].str.split(
        '-').str[0].astype(int), m['年月日'].str.split('-').str[1].astype(int)
    # bic.dtypes
    m = mymerge(m, bic)
    del m['年'], m['月'], bic['年'], bic['月']
    m.年月日 = pd.to_datetime(m.年月日, format='%Y-%m-%d').apply(lambda x: x.date())

    return m





def timeSpan(func):
    def wrapper(*args, **kw):
        print('call %s():' % func.__name__)
        start = time.time()
        x = func(*args, **kw)
        end = time.time()
        print('Complete in {} second(s)'.format(end-start))
        return x
    return wrapper


def tech(m):
    
    # return
    @timeSpan
    def Return(df, cols, periods):
        for col in cols:
            for days in periods:
                n = 240 / days
                df['r' + str(days) + '.' + col] = (df[col].shift(-days) / df[col]) ** n - 1


    Return(m, ['調整收盤價'], [5, 10, 20, 40, 60, 120])


    # log return
    @timeSpan
    def lnReturn(df, cols, periods):
        for col in cols:
            for days in periods:
                n = 240 / days
                df['lnr' + str(days) + '.' + col] = np.log(df[col].shift(-days) / df[col]) * n


    lnReturn(m, ['調整收盤價'], [5, 10, 20, 40, 60, 120])


    @timeSpan
    def Normalize(df, cols):
        for col in cols:
            df[col + '.nmz'] = (df[col] - df[col].mean()) / df[col].std()


    Normalize(m, ['r5.調整收盤價', 'r10.調整收盤價', 'r20.調整收盤價', 'r40.調整收盤價', 'r60.調整收盤價', 'r120.調整收盤價'])

    # rsi
    m['ch'] = m['調整收盤價'].diff()
    m['ch_u'], m['ch_d'] = m['ch'], m['ch']
    m.ix[m.ch_u < 0, 'ch_u'] = 0
    m.ix[m.ch_d > 0, 'ch_d'] = 0
    m['ch_d'] = m['ch_d'].abs()

    # default: adjust=True, see formula https://github.com/pandas-dev/pandas/issues/8861, when adjust=false, see formula http://www.fmlabs.com/reference/default.htm?url=RSI.htm
    m['rsi'] = m.ch_u.ewm(alpha=1 / 14).mean() / (
                m.ch_u.ewm(alpha=1 / 14).mean() + m.ch_d.ewm(alpha=1 / 14).mean()) * 100  # 與r和凱基同
    m = m.drop(['ch', 'ch_u', 'ch_d'], axis=1)


    # ma
    @timeSpan
    def ma(*period):
        for n in period:
            m['MA' + str(n)] = m['收盤價'].rolling(window=n).mean()


    @timeSpan
    def ma_adj(*period):
        for n in period:
            m['MA' + str(n) + '.adj'] = m['調整收盤價'].rolling(window=n).mean()


    ma(5, 10, 20, 60, 120)
    ma_adj(5, 10, 20, 60, 120)

    # price
    m['price'] = (m['最高價'] + m['最低價'] + 2 * m['收盤價']) / 4
    m['price.adj'] = (m['調整最高價'] + m['調整最低價'] + 2 * m['調整收盤價']) / 4

    # macd, osc
    m['EMA12'] = m['price'].ewm(alpha=2 / 13).mean()
    m['EMA26'] = m['price'].ewm(alpha=2 / 27).mean()
    m['DIF'] = m['EMA12'] - m['EMA26']
    m['MACD'] = m.DIF.ewm(alpha=0.2).mean()
    m['MACD1'] = (m['EMA12'] - m['EMA26']) / m['EMA26'] * 100
    m['OSC'] = m.DIF - m.MACD


    # stdev
    def stdev(df, cols, periods):
        for col in cols:
            for p in periods:
                df[f'{col}:stdev{p}'] = df[col].rolling(window=p).std()
        return df


    m = stdev(m, ['price', 'price.adj'], [5, 10, 20])


    # bband
    m['std20'] = m['price'].rolling(window=20).std()
    m['mavg'] = m['price'].rolling(window=20).mean()
    m['up'] = m.mavg + m['std20'] * 2
    m['dn'] = m.mavg - m['std20'] * 2
    m['bband'] = (m['收盤價'] - m.mavg) / m['std20']

    # bband adj
    m['std20.adj'] = m['price.adj'].rolling(window=20).std()
    m['mavg.adj'] = m['price.adj'].rolling(window=20).mean()
    m['up.adj'] = m['mavg.adj'] + m['std20.adj'] * 2
    m['dn.adj'] = m['mavg.adj'] - m['std20.adj'] * 2
    m['bband.adj'] = (m['調整收盤價'] - m['mavg.adj']) / m['std20.adj']

    # kd
    m['max9'] = m['最高價'].rolling(window=9).max()
    m['min9'] = m['最低價'].rolling(window=9).min()
    m['rsv'] = (m['收盤價'] - m.min9) / (m.max9 - m.min9)
    m['k'] = m.rsv.ewm(alpha=1 / 3).mean()
    m['d'] = m.k.ewm(alpha=1 / 3).mean()

    # others
    m['high-low'] = (m['最高價'] - m['最低價']) / m['收盤價']
    m['pch'] = (m['收盤價'] - m['收盤價'].shift()) / m['收盤價'].shift()
    m['pctB'] = (m['price'] - m.dn) / (m.up - m.dn)
    m['close-up'] = (m['收盤價'] - m.up) / (m['price'].rolling(window=20).std() * 2)
    m['close-dn'] = (m['收盤價'] - m.dn) / (m['price'].rolling(window=20).std() * 2)

    m['pctB.adj'] = (m['price.adj'] - m['dn.adj']) / (m['up.adj'] - m['dn.adj'])
    m['close-up.adj'] = (m['調整收盤價'] - m['up.adj']) / (m['price.adj'].rolling(window=20).std() * 2)
    m['close-dn.adj'] = (m['調整收盤價'] - m['dn.adj']) / (m['price.adj'].rolling(window=20).std() * 2)

    timeDelta('before trend')


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
            df1['trend_{}'.format(col)] = np.sign(df1['pch_{}'.format(col)])
            i = df1[df1['trend_{}'.format(col)] == 0].index
            if len(i) > 0:
                df1.ix[i, 'trend_{}'.format(col)] = df1.ix[i - 1, 'trend_{}'.format(col)].tolist()
        return df1



    @timeSpan
    def reversion(df, columns):
        df1 = deepcopy(df)
        for col in columns:
            df1['reversion_{}'.format(col)] = df1['trend_{}'.format(col)].diff()/2
        return df1

    m = pch(m, ['收盤價', 'MA5', 'MA10', 'MA20', 'MA60', 'MA120'])
    m = trend(m, ['收盤價', 'MA5', 'MA10', 'MA20', 'MA60', 'MA120'])
    m = reversion(m, ['收盤價', 'MA5', 'MA10', 'MA20', 'MA60', 'MA120'])

    @timeSpan
    def local_min_or_max(df, columns):
        df1 = deepcopy(df)
        for col in columns:
            #init
            df1['local_min(max)_{}'.format(col)] = df1['reversion_{}'.format(col)] - df1['reversion_{}'.format(col)]
            
            i = df1.ix[df1['reversion_{}'.format(col)] == 1].index
            # local min
            df1.ix[i - 1, 'local_min(max)_{}'.format(col)] = -1
            
            i = df1.ix[df1['reversion_{}'.format(col)] == -1].index
            # local max
            df1.ix[i - 1, 'local_min(max)_{}'.format(col)] = 1
        return df1



    m = local_min_or_max(m, ['收盤價', 'MA5', 'MA10', 'MA20', 'MA60', 'MA120'])

    @timeSpan
    def new_high_or_low(df, columns):
        df1 = deepcopy(df)
        for col in columns:
            df1['new_high(low)_{}'.format(col)] = df1['local_min(max)_{}'.format(col)] - df1['local_min(max)_{}'.format(col)]
            i = df1.ix[df1['local_min(max)_{}'.format(col)] == 1, 'local_min(max)_{}'.format(col)].index

            l = (df1['{}'.format(col)][i] - df1['{}'.format(col)][i].shift()).tolist()
            ii = np.array([i for i, j in enumerate(l) if j > 0])
            if len(ii) > 0:
                df1.ix[i[ii], 'new_high(low)_{}'.format(col)] = 1
            i = df1.ix[df1['local_min(max)_{}'.format(col)] == -1, 'local_min(max)_{}'.format(col)].index

            l = (df1['{}'.format(col)][i] - df1['{}'.format(col)][i].shift()).tolist()
            ii = np.array([i for i, j in enumerate(l) if j < 0])
            if len(ii) > 0:
                df1.ix[i[ii], 'new_high(low)_{}'.format(col)] = -1
        return df1

    m = new_high_or_low(m, ['收盤價', 'MA5', 'MA10', 'MA20', 'MA60', 'MA120'])

    list(m)

    m[['local_min(max)_MA5', 'new_high(low)_MA5']].head(100)

    m['span'] = abs(m['調整收盤價']-m.調整開盤價)/m['調整收盤價']
    m['span_high-low'] = abs(m['調整最高價']-m['調整最低價'])/m['調整收盤價']
    m['upperShadow'] = (m['調整最高價'] - m[['調整開盤價', '調整收盤價']].max(axis=1))/m['調整收盤價']
    m['lowerShadow'] = (m[['調整開盤價', '調整收盤價']].min(axis=1) - m['調整最低價'])/m['調整收盤價']
    m['upperShadow/span'] =m['upperShadow']/(m['span']+0.1**10*m['調整收盤價'])
    m['lowerShadow/span'] =m['lowerShadow']/(m['span']+0.1**10*m['調整收盤價'])
    # m['span/upperShadow'] =m['span']/m['upperShadow']
    # m['span/lowerShadow'] =m['span']/m['lowerShadow']
    m['span/(high-low)'] =m['span']/m['span_high-low']
    del m['d']
    m['high-low_1ag1'] = m['high-low'].shift()
    m['high-low_lag2'] = m['high-low'].shift(2)
    m['upperShadow_lag1'] = m['upperShadow'].shift()
    m['lowerShadow_lag1'] = m['lowerShadow'].shift()
    m['upperShadow/span_lag1'] = m['upperShadow/span'].shift()
    m['lowerShadow/span_lag1'] = m['lowerShadow/span'].shift()
    # m['span/upperShadow_lag1'] = m['span/upperShadow'].shift()
    # m['span/lowerShadow_lag1'] = m['span/lowerShadow'].shift()
    m['spandiff'] = m.span.diff()
    m['spanudiff'] = m[['調整開盤價', '調整收盤價']].max(axis=1).diff()
    m['spanldiff'] = m[['調整開盤價', '調整收盤價']].min(axis=1).diff()
    m['span/(high-low)_lag1'] = m['span/(high-low)'].shift()

    timeDelta('before OSCsign')

    m['OSCsign'] = np.sign(m.OSC)
    m['OSCgroup'] = 0

    OSCsign = m['OSCsign'].tolist()
    gr = m['OSCgroup'].tolist()
    g = 0
    for i in range(len(OSCsign)-1):
        if OSCsign[i]*OSCsign[i+1] < 0:
            g+=1
            gr[i+1] = g
        else:
            gr[i+1] = g

    m['OSCsign'], m['OSCgroup'] = OSCsign, gr
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

    grouped = m.groupby('OSCgroup')
    l = grouped['OSC'].apply(minORmax).tolist()

    d = {}
    for i, v in enumerate(l):
        d[i+2] = v
    d[0], d[1] = np.nan, np.nan

    # previous high/low
    m[['gr1']] = m[['OSCgroup']].applymap(lambda x:d[x])

    m['change'] = 0
    @timeSpan
    def OSCbreakpoint(df):
        df1 = deepcopy(df)
        df1=df1.reset_index(drop=True)  # without this df1.ix[0,'gr1'] is only defined in first group
        if df1['OSC'].max()>0:
            for i in range(len(df1['gr1'])):
                if df1.ix[i,'OSC']>df1.ix[i,'gr1']:
                    df1.ix[i, 'change'] = 1
                    break
            return df1
        if df1['OSC'].min()<0:
            for i in range(len(df1['gr1'])):
                if df1.ix[i,'OSC']<df1.ix[i,'gr1']:
                    df1.ix[i, 'change'] = -1
                    break
            return df1
        else:
            return df1

    m = grouped.apply(OSCbreakpoint).reset_index(drop=True)
    del m['OSCsign'], m['OSCgroup'], m['gr1']

    timeDelta('m')

    # m = mymerge(m, index).sort_values(['年月日'])
    m['漲跌(+/-)'] = m['漲跌(+/-)'].replace('＋', 1).replace('－', -1).replace('X', 0).replace(' ', None).astype(float)
    m['外資鉅額交易'] = m['外資鉅額交易'].replace('yes', 1).replace('no', 0).astype(float)
    m['投信鉅額交易'] = m['投信鉅額交易'].replace('yes', 1).replace('no', 0).astype(float)
    m = m.drop(['證券名稱', '公司名稱'], 1)

    m = m.drop_duplicates(['年月日'])
    m = m.dropna(axis=1, how='all')
    m = m.dropna(subset=['證券代號'])
    m = m[~pd.isnull(m['年月日'])]
    m = m.reset_index(drop=True)

    return m


def save(conn,table, m):
    date_cols = ['年月日']
    varchar_cols = ['證券代號']
    float_cols = [col for col in list(m) if col not in date_cols + varchar_cols]
    m = ast.to_float(float_cols, m)
    columns = date_cols + varchar_cols + float_cols
    fieldTypes = ['date' for col in date_cols] + ['varchar(14)' for col in varchar_cols] + ['real' for col in float_cols]
    pkeys = ['年月日', '證券代號']
    sqlc.ct_pg(conn, table, columns, fieldTypes, pkeys)
    dftosql.i_pg_batch(conn, table, m)


start = time.time()

def timeDelta(s):
    global start
    end = time.time()
    print(s, 'timedelta: ', end - start)
    start = end


mysum = conn_local_pg('mysum')
table = 'mysums'
sqlc.dropTablePostgre(table, mysum)

summary = conn_local_pg('summary')
mops = conn_local_pg('mops')
tse = conn_local_pg('tse')
bi = conn_local_pg('bic')

def run(summary, mops, tse, bi, mysum, id):

    ac = account(summary, id)
    fin = finance(summary, id)

    # timeDelta('summary')

    inc = income(mops, id)
    bal = balance(mops, id)

    re = report(inc, bal)

    # timeDelta('mops')

    tsed = tsedata(tse, id)

    # timeDelta('tse')

    m = merge(tsed, ac, re)

    # timeDelta('merge')

    bic = businese(bi)

    m = mergebic(m, bic)

    m = tech(m)

    # timeDelta('tech')

    # return m

    save(mysum, table, m)


def main(coll):  
    begin = time.time()

    df = select.sd(['證券代號'], 'mysum').df(tse).dropna()
    ids = list(df['證券代號'])

    errs = []
    for id in ids:
        try:
            now = time.time()
            print(id, now-begin)
            run(summary, mops, tse, bi, mysum, id)
        except Exception as e:
            d = {'id':id, 'e':e}
            print(d)
            errs.append(d)
            coll.insert_one({'id':id, 'e':str(e)})

    return errs


port = int(os.getenv('MONGO_DOCKER_PORT'))
user = os.getenv('MONGO_DOCKER_USER')
pwd = os.getenv('MONGO_DOCKER_PWD')
client = MongoClient('localhost', port, username=user, password=pwd)

mysums = client['mysums']
coll = mysums['err']

errs = main(coll)
print(errs)

print('finish', time.time() - start)


# id = '3665'
# df = run(summary, mops, tse, bi, mysum, id)
# df = select.sd(['證券代號'], 'mysum').df(tse).dropna()
# ids = list(df['證券代號'])
# print(ids)