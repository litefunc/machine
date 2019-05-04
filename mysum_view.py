from sql.pg import select, insert, delete
from common.connection import conn_local_pg
import syspath
import pandas as pd
import numpy as np
import cytoolz.curried
import datetime as dt
import os
import sys
if os.getenv('MY_PYTHON_PKG') not in sys.path:
    sys.path.append(os.getenv('MY_PYTHON_PKG'))


tse = conn_local_pg('tse')
cur = tse.cursor()

table1 = 'compute'
table2 = '個股日本益比、殖利率及股價淨值比'
table3 = '每日收盤行情(全部(不含權證、牛熊證))'
df1 = select.saw(table1, {'證券代號': '5522'}).df(tse)
list(df1)
df2 = select.saw('每日收盤行情(全部(不含權證、牛熊證))', {'證券代號': '5522'}).df(tse)
list(df2)
df3 = select.sw(['年月日', '證券代號', '殖利率(%%)', '本益比', '股價淨值比'],
                table2, {'證券代號': '5522'}).df(tse)
list(df3)
cols1 = ', '.join([f'a."{col}"' for col in list(df1)])
cols2 = ', '.join([f'b."{col}"' for col in ['殖利率(%)', '本益比', '股價淨值比']])
cols3 = ', '.join([f'c."{col}"' for col in [
                  '證券名稱', '開盤價', '最高價', '最低價', '收盤價', '成交股數', '成交筆數', '成交金額']])

sql = f'''create or replace view mysum as select {cols1}, {cols2}, {cols3} from "{table1}" as a full outer join "{table2}" as b on a."年月日"=b."年月日" and a."證券代號"=b."證券代號" full outer join "{table3}" as c on a."年月日"=c."年月日" and a."證券代號"=c."證券代號"'''


cur.execute(sql)

tse.commit()

df4 = select.saw('mysum', {'證券代號': '5522'}).df(tse)
list(df4)
