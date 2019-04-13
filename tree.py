import time
import datetime as dt
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from graphviz import Source
from bokeh.plotting import figure, show
from sklearn import ensemble
from bokeh.plotting import figure, show, output_file
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn import svm
import pandas as pd
import numpy as np
import os
import sys
import sqlCommand as sqlc
from common.connection import conn_local_lite, conn_local_pg
import syspath

if os.getenv('MY_PYTHON_PKG') not in sys.path:
    sys.path.append(os.getenv('MY_PYTHON_PKG'))


MY_PYTHON_PROJECT = os.getenv('MY_PYTHON_PROJECT')
if MY_PYTHON_PROJECT not in sys.path:
    sys.path.append(MY_PYTHON_PROJECT)

os.chdir(f'{MY_PYTHON_PROJECT}/machine')

conn_pg = conn_local_pg('mysum')

df = pd.read_sql_query('select * from mysum', conn_pg)

print(df.dtypes)
cols = list(df)
for col in cols:
    print(col)

ycol = 'lnr20.調整收盤價'
rcol = ['{}.調整收盤價'.format(col) for col in ['r5', 'r10', 'r20', 'r40',
                                           'r60', 'r120', 'lnr5', 'lnr10', 'lnr20', 'lnr40', 'lnr60', 'lnr120']]
rcol_normalize = ['r5.調整收盤價.nmz', 'r10.調整收盤價.nmz', 'r20.調整收盤價.nmz',
                  'r40.調整收盤價.nmz', 'r60.調整收盤價.nmz', 'r120.調整收盤價.nmz']
objcol = ['年月日']
dropcol = objcol + [col for col in rcol if col != ycol]
df1 = df.drop(dropcol, 1)
cols = list(df1)
xcol = [col for col in cols if col != ycol]
df1 = df1[~pd.isnull(df1[ycol])][[ycol] + xcol].replace([np.inf, -
                                                         np.inf], np.nan).fillna(0).reset_index(drop=True)

x_train, x_test, y_train, y_test = train_test_split(
    df1[xcol], df1[[ycol]], test_size=0.1, random_state=0)

# ----tree----

# cv = KFold(n=x.shape[0], n_folds=10, shuffle=True)
n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True)

alphas, mses, R2s = [], [], []
for i in range(1, 100, 2):
    alpha = i/1000
    params = {'min_impurity_decrease': alpha}     # complexity parameter
    clf = tree.DecisionTreeRegressor(**params)
    mse = np.mean(-cross_val_score(clf, x_train, y_train, cv=kf,
                                   scoring='neg_mean_squared_error'))  # sklearn put '-' in front of mse
    R2 = 1-mse/y_train.var()
    print('min_impurity_decrease : %.3f | mse : %.3f | R2 : %.3f' %
          (alpha/1000, mse, R2))
    alphas.append(alpha/1000)
    mses.append(mse)
    R2s.append(R2)
print('i:', np.argmax(R2s), 'min_impurity_decrease:', alphas[np.argmax(
    R2s)], 'mse:', mses[np.argmax(R2s)], 'R2:', R2s[np.argmax(R2s)])
params = {'min_impurity_decrease': alphas[np.argmax(R2s)]}


clf = tree.DecisionTreeRegressor(**params)
clf.fit(x_train, y_train)
print('clf.tree_.node_count : ', clf.tree_.node_count, 'nrow : ', len(df))
r2_tree = clf.score(x_test, y_test)
print('test R2 : ', r2_tree)

feature_importances = {}
for i in range(len(list(x_train))):
    feature_importances[list(x_train)[i]] = clf.feature_importances_[i]
for i in reversed(sorted(feature_importances.items(), key=lambda x: x[1])):
    print(i)

clf.predict(np.array(df1[xcol])[-1].reshape(1, -1))

now = dt.datetime.now()
# ":" is reserved character in windows
f = '~/Public/share/tree/tree.{0}.{1}.{2} {3} {4}h{5}m'.format(df['證券代號'][0], ycol, 'Rsq-'+str(
    format(round(r2_tree, 5), '.5g')), now.date(), now.hour, now.minute)

filename = os.path.expanduser(f)

dot_data = tree.export_graphviz(
    clf, out_file=filename, feature_names=xcol, filled=True, rounded=True, special_characters=True)

file = open(filename, 'r')  # READING DOT FILE
text = file.read()
src = Source(text)
src.format = 'svg'
#  save it to svg and open with browser will be much faster
src.render(filename, view=True)