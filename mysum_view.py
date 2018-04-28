import pandas as pd
import numpy as np
import cytoolz.curried
import datetime as dt
import os
import sys
if os.getenv('MY_PYTHON_PKG') not in sys.path:
    sys.path.append(os.getenv('MY_PYTHON_PKG'))
import syspath

from common.connection import conn_local_pg
from sql.pg import select, insert, delete

tse = conn_local_pg('tse')
cur = tse.cursor()

table1 = 'compute'
table2 = '個股日本益比、殖利率及股價淨值比'
table3 = '每日收盤行情(全部(不含權證、牛熊證))'
df1 = select.saw(table1, {'證券代號':'5522'}).df(tse)
list(df1)
df2 = select.saw('每日收盤行情(全部(不含權證、牛熊證))', {'證券代號':'5522'}).df(tse)
list(df2)
df3 = select.sw(['年月日', '證券代號', '殖利率(%%)', '本益比', '股價淨值比'], table2, {'證券代號':'5522'}).df(tse)
list(df3)
cols1=', '.join([f'a."{col}"' for col in list(df1)])
cols2=', '.join([f'b."{col}"' for col in ['殖利率(%)', '本益比', '股價淨值比']])
cols3=', '.join([f'c."{col}"' for col in ['證券名稱', '開盤價', '最高價', '最低價', '收盤價', '成交股數', '成交筆數', '成交金額']])

sql = f'''create view mysum as select {cols1}, {cols2}, {cols3} from "{table1}" as a full outer join "{table2}" as b on a."年月日"=b."年月日" and a."證券代號"=b."證券代號" full outer join "{table3}" as c on a."年月日"=c."年月日" and a."證券代號"=c."證券代號"'''


sql = '''
create view mysum as select a."年月日", a."證券代號", c."證券名稱", c."開盤價", c."最高價", c."最低價", c."收盤價", a."收盤價:調整", a."開盤價:調整", a."最高價:調整", a."最低價:調整", 
a."price", a."price:調整", c."成交股數", c."成交筆數", c."成交金額", b."殖利率(%)", b."本益比", b."股價淨值比",
 a."DIF", a."MACD", a."MACD1", a."OSC", a."Avg_Band", a."Upper_Band", a."Lower_Band", a."(price-Avg_Band)/price:標準差20", a."Avg_Band:調整", a."Upper_Band:調整",
  a."Lower_Band:調整", a."(price:調整-Avg_Band:調整)/price:調整:標準差20", a."報酬率5", a."報酬率10", a."報酬率20", a."報酬率40", a."報酬率60", a."報酬率120", a."報酬率5:調整",
   a."報酬率10:調整", a."報酬率20:調整", a."報酬率40:調整", a."報酬率60:調整", a."報酬率120:調整", a."ln報酬率5", a."ln報酬率10", a."ln報酬率20", a."ln報酬率40", a."ln報酬率60",
    a."ln報酬率120", a."ln報酬率5:調整", a."ln報酬率10:調整", a."ln報酬率20:調整", a."ln報酬率40:調整", a."ln報酬率60:調整", a."ln報酬率120:調整", a."MA5", a."MA10", a."MA20",
     a."MA40", a."MA60", a."MA120", a."MA5:調整", a."MA10:調整", a."MA20:調整", a."MA40:調整", a."MA60:調整", a."MA120:調整", a."RSI", a."K", a."D", a."J" from "compute" as a 
      full outer join "個股日本益比、殖利率及股價淨值比" as b on a."年月日"=b."年月日" and a."證券代號"=b."證券代號" 
      full outer join "每日收盤行情(全部(不含權證、牛熊證))" as c on a."年月日"=c."年月日" and a."證券代號"=c."證券代號"
'''
cur.execute(sql)

df4= select.saw('mysum', {'證券代號':'5522'}).df(tse)
list(df4)