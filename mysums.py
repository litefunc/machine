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

def merge(tse, ac, report):
    df = cytoolz.reduce(mymerge, [tse, ac, report])
    # should convert to datetime before sort, or the result is wrong
    df.年月日 = pd.to_datetime(
        df.年月日, format='%Y/%m/%d').apply(lambda x: x.date())
    df = df.sort_values(['年月日', '證券代號']).reset_index(
        drop=True)  # reset_index make the index ascending
    return df


start = time.time()


def timeDelta(s):
    global start
    end = time.time()
    print(s, 'timedelta: ', end - start)
    start = end


def timeSpan(func):
    def wrapper(*args, **kw):
        print('call %s():' % func.__name__)
        start = time.time()
        x = func(*args, **kw)
        end = time.time()
        print('Complete in {} second(s)'.format(end-start))
        return x
    return wrapper


id = '5522'

summary = conn_local_pg('summary')

ac = account(summary, id)
fin = finance(summary, id)

timeDelta('summary')

mops = conn_local_pg('mops')
cur_mops = mops.cursor()

inc = income(mops, id)
bal = balance(mops, id)

re = report(inc, bal)

timeDelta('mops')

tse = conn_local_pg('tse')

tsed = tsedata(tse, id)

timeDelta('tse')

m = merge(tsed, ac, re)

timeDelta('merge')
