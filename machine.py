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
# from pyearth import Earth
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

# ----mars----
# Fit an Earth model
model = Earth()
model.fit(x_train, y_train)

# Print the model
print(model.trace())
print(model.summary())


# Plot the model
y_hat = model.predict(x_train)
pyplot.figure()
pyplot.plot(x_train.ix[:, 6], y_train, 'r.')
pyplot.plot(x_train.ix[:, 6], y_hat, 'b.')
pyplot.xlabel('x_6')
pyplot.ylabel('y')
pyplot.title('Simple Earth Example')
pyplot.show()
error = np.array(y_test) - model.predict(np.array(x_test))
r2_mars = model.score(x_test, y_test)
print('test R2 : ', r2_mars)
model.predict([np.array(df1[xcol])[-1]])


# ----Gradient Tree Boosting----

params = {'n_estimators': 1000, 'max_leaf_nodes': 8, 'subsample': 0.5,
          'learning_rate': 0.01, 'min_samples_leaf': 1, 'verbose': 1, 'min_impurity_decrease': 0.05}

gbt = ensemble.GradientBoostingRegressor(**params)
gbt.fit(x_train, y_train)
r2_gbm = gbt.score(x_test, y_test)
print('test R2 : ', r2_gbm)

# negative cumulative sum of oob improvements
cumsum = np.cumsum(gbt.oob_improvement_)

# best n_estimators base on oob improvements, 'subsample' must <1
oob_best_iter = np.argmax(cumsum)
print('oob_best_iter', oob_best_iter)

# #############################################################################
# Plot training deviance

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(x_test)):
    test_score[i] = clf.loss_(np.array(y_test), y_pred)

pyplot.figure(figsize=(12, 6))
pyplot.subplot(1, 2, 1)
pyplot.title('Deviance')
pyplot.plot(np.arange(params['n_estimators']) + 1, gbt.train_score_, 'b-',
            label='Training Set Deviance')
pyplot.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
            label='Test Set Deviance')
pyplot.legend(loc='upper right')
pyplot.xlabel('Boosting Iterations')
pyplot.ylabel('Deviance')

# #############################################################################
# Plot feature_importance


feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
factors = np.array(list(x_train))[sorted_idx].tolist()
x = feature_importance[sorted_idx].tolist()

p1 = figure(title="feature_importance", plot_width=1200,
            plot_height=6000, y_range=factors, x_range=[min(x), max(x)])

p1.segment(0, factors, x, factors, line_width=2)

# output_file("feature_importance.html")
show(p1)
# show(vplot(p1, p2))  # open a browser

# plot partial dependence
sorted_idx = np.argsort(-feature_importance)
factors = np.array(list(x_train))[sorted_idx].tolist()
features = sorted_idx[0:6]

fig, axs = plot_partial_dependence(clf, x_train, features,
                                   feature_names=list(x_train),
                                   n_jobs=3, grid_resolution=100)
fig.suptitle('Partial dependence of {}'.format(ycol))
plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle

fig = plt.figure()

target_feature = features[0:2]
pdp, axes = partial_dependence(clf, target_feature,
                               X=x_train, grid_resolution=50)
XX, YY = np.meshgrid(axes[0], axes[1])
Z = pdp[0].reshape(list(map(np.size, axes))).T
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
                       cmap=plt.cm.BuPu, edgecolor='k')
ax.set_xlabel(list(x_train)[target_feature[0]])
ax.set_ylabel(list(x_train)[target_feature[1]])
ax.set_zlabel('Partial dependence')
#  pretty init view
ax.view_init(elev=22, azim=122)
plt.colorbar(surf)
plt.suptitle('Partial dependence of {}'.format(ycol))
plt.subplots_adjust(top=0.9)

plt.show()

# - plot time series--
df.dtypes

#df2 = df[['年月日', 'lnr20', 'r20Std', 'lnmo']]
p1 = figure(x_axis_type="datetime", plot_width=1800,
            plot_height=800, title='compared to {}'.format(ycol))
p1.grid.grid_line_alpha = 0.3
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'value'

circle_size = 1
p1.circle(df['年月日'], df[ycol], color="#d62728", legend=ycol, size=circle_size)
p1.circle(df['年月日'], df[factors[0]], color='#1f77b4',
          legend=factors[0], size=circle_size)
p1.circle(df['年月日'], df[factors[1]], color="#2ca02c",
          legend=factors[1], size=circle_size)
p1.circle(df['年月日'], df[factors[2]], color="#ff7f0e",
          legend=factors[2], size=circle_size)
p1.circle(df['年月日'], df[factors[3]], color="#9467bd",
          legend=factors[3], size=circle_size)
p1.circle(df['年月日'], df[factors[4]], color="#8c564b",
          legend=factors[4], size=circle_size)
p1.circle(df['年月日'], df[factors[5]], color="#e377c2",
          legend=factors[5], size=circle_size)

p1.legend.location = "top_left"
#output_file("stocks.html", title="stocks.py example")
show(p1)


# ----lasso----

model_bic = LassoLarsIC(criterion='bic', verbose=True)
t1 = time.time()
model_bic.fit(x_train, y_train)
t_bic = time.time() - t1
alpha_bic_ = model_bic.alpha_
model_bic.score(x_test, y_test)

model_aic = LassoLarsIC(criterion='aic')
model_aic.fit(x_train, y_train)
alpha_aic_ = model_aic.alpha_
model_aic.score(x_test, y_test)

model_cv = LassoCV(verbose=True)
model_cv.fit(x_train, y_train)
alpha_cv_ = model_cv.alpha_
model_cv.score(x_test, y_test)

model_Larscv = LassoLarsCV(verbose=True)
model_Larscv.fit(x_train, y_train)
alpha_Larscv_ = model_Larscv.alpha_
model_Larscv.score(x_test, y_test)


def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_
    alphas_ = model.alphas_
    criterion_ = model.criterion_
    pyplot.plot(-alphas_, criterion_, '--', color=color,
                label='%s criterion' % name)
    pyplot.axvline(-alpha_, color=color, label='alpha: %s estimate' % name)
    pyplot.xlabel('-alpha')
    pyplot.ylabel('criterion')


pyplot.figure()
plot_ic_criterion(model_aic, 'AIC', 'b')
plot_ic_criterion(model_bic, 'BIC', 'r')
pyplot.legend()
pyplot.title(
    'Information-criterion for model selection (training time %.3fs)' % t_bic)
pyplot.tight_layout()
pyplot.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)


# ----SVR----

clf = svm.SVR(kernel='linear')  # extremely slow
clf.fit(x_train, np.array([i for i in y_train[ycol]]))
clf.score(x_test, y_test)
clf.predict(np.array(x)[-1].reshape(1, -1))
