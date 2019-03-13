# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tushare as ts
import scipy.optimize as sco
import scipy.interpolate as sci
from scipy import stats

# thanks  https://tushare.pro 提供数据
api = ts.pro_api('d015d236481abd2223eb9bf6a2d23ff0a05fdea9fd9dd231a717915a')  # 设置API

df = pd.DataFrame()
df = ts.pro_bar(pro_api=api, ts_code='000651.SZ', adj='qfq', start_date='20140528', end_date='20170526') # 获得前复权
s000651 = df['close'] # 获得收盘价
s000651.name = '000651' # 格力电器

df = ts.pro_bar(pro_api=api, ts_code='600519.SH', adj='qfq', start_date='20140528', end_date='20170526') # 获得前复权
s600519 = df['close'] # 获得收盘价
s600519.name = '600519' # 贵州茅台
df = ts.pro_bar(pro_api=api, ts_code='601318.SH', adj='qfq', start_date='20140528', end_date='20170526') # 获得前复权
s601318 = df['close'] # 获得收盘价
s601318.name = '601318' # 中国平安


data = pd.DataFrame({'000651':s000651, '600519':s600519, '601318':s601318})

#data.index.name = 'date' #必须补充一下index的名字
data = data.dropna()




print(data.info())

data.to_excel('stock_data2.xlsx')

data = pd.read_excel('stock_data2.xlsx')
#data.index = data['date'].tolist() #将date列拷贝一份，设为index列
#data.pop('date') #移除date列，因为date列已经成为index列了，以后只用看index

data.plot(figsize=(10, 8), grid = True)

#计算对数收益率。金融计算收益率的时候大部分用对数收益率 (Log Return) 而不是用算数收益率
log_returns = np.log(data / data.shift(1))
# 画出每只股票收益率的直方图，了解一下分布情况。
log_returns.hist(bins=50, figsize=(12, 9))



#我们一共有10支股票
number_of_assets = 3
#生成10个随机数
weights = np.random.random(number_of_assets)
#将10个随机数归一化，每一份就是权重，权重之和为1
weights /= np.sum(weights)

tech_rets = data.pct_change()
rets = tech_rets.dropna()

"""
portfolio_returns = []
portfolio_volatilities = []
for p in range (5000):
    weights = np.random.random(number_of_assets)
    weights /= np.sum(weights)
    portfolio_returns.append(np.sum(rets.mean() * weights) * 252)
    portfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights))))
    
    
    
print('hello')
plt.figure(figsize=(9, 5)) #作图大小
plt.scatter(portfolio_volatilities, portfolio_returns, c=portfolio_returns / portfolio_volatilities, marker='o') #画散点图
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
"""

def statistics(weights):        
    #根据权重，计算资产组合收益率/波动率/夏普率。
    #输入参数
    #==========
    #weights : array-like 权重数组
    #权重为股票组合中不同股票的权重    
    #返回值
    #=======
    #pret : float
    #      投资组合收益率
    #pvol : float
    #      投资组合波动率
    #pret / pvol : float
    #    夏普率，为组合收益率除以波动率，此处不涉及无风险收益率资产
    #

    weights = np.array(weights)
    pret = np.sum(rets.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    return np.array([pret, pvol, pret / pvol])

def min_func_sharpe(weights):
    return -statistics(weights)[2]

def min_func_variance(weights):
    return statistics(weights)[1] ** 2

def min_func_port(weights):
    return statistics(weights)[1]

target_returns = np.linspace(0.35, 0.65, 30)
target_volatilities = []
for tret in target_returns:
    cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - tret},
            {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
    res = sco.minimize(min_func_port, number_of_assets * [1. / number_of_assets,], method='SLSQP', bounds=bnds, constraints=cons)
    target_volatilities.append(res['fun'])

#画散点图
plt.figure(figsize=(9, 5))
#圆点为随机资产组合
plt.scatter(portfolio_volatilities, portfolio_returns,
            c=portfolio_returns / portfolio_volatilities, marker='o')
#叉叉为有效边界            
plt.scatter(target_volatilities, target_returns,
            c=target_returns / target_volatilities, marker='x')
#红星为夏普率最大值的资产组合            
plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0],
         'r*', markersize=15.0)
#黄星为最小方差的资产组合            
plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0],
         'y*', markersize=15.0)
            # minimum variance portfolio
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')