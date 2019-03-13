import pandas as pd  
import numpy as np
import quandl   # 获取股票数据

from datetime import date
import matplotlib.pyplot as plt

quandl.ApiConfig.api_key = "M_ryjzs5imL2YeBqhfut"

# 创建空的DataFrame变量，用于存储股票数据
StockPrices = pd.DataFrame()

# 设置股票数据的开始和结束的时间
start = date(2015,12,31)
end = date(2017,12,31) #最多只能到2018年3月27日  4月11日后不再维护这个数据

# 创建股票代码的列表
ticker_list = ['AAPL', 'MSFT', 'XOM', 'JNJ', 'JPM', 'AMZN', 'GE', 'FB', 'T']
#苹果 微软 埃克森美孚 强生 摩根 亚马逊 通用电气 脸书 AT&T

# 使用循环，挨个获取每只股票的数据，并存储调整后的收盘价
for ticker in ticker_list:
    data = quandl.get('WIKI/'+ticker, start_date=start, end_date=end)
    StockPrices[ticker] = data['Adj. Close']  # 注意 Adj. 和 Close 之间有一空格
# 股票的数据都存在StockPrices里了

# 输出数据的前5行
# Print.()
# StockPrices.plot()
StockPrices.to_excel('StockPrices.xlsx')
StockPrices.dropna().plot()#
plt.show()

# 计算每日收益率，并丢弃缺失值
# pct_change() 用于计算同colnums两个相邻的数字之间的变化率
StockReturns = StockPrices.pct_change().dropna()
StockReturns.to_excel('StockReturns.xlsx')

# 打印前5行数据
#print(StockReturns.head())


# 随便给一个投资组合
# 定义一个基金权重
# 设置组合权重，存储为numpy数组类型
portfolio_weights = np.array([0.12, 0.15, 0.08, 0.05, 0.09, 0.10, 0.11, 0.14, 0.16])

# 将收益率数据拷贝到新的变量 stock_return 中，这是为了后续调用的方便
stock_return = StockReturns.copy()

# 计算加权的股票收益
WeightedReturns = stock_return.mul(portfolio_weights, axis=1)

# 计算投资组合的收益
StockReturns['Portfolio'] = WeightedReturns.sum(axis=1)

# 绘制组合收益随时间变化的图
# StockReturns.Portfolio.plot()
# plt.show()


# 累积收益曲线绘制函数
def cumulative_returns_plot(name_list):
    for name in name_list:
        CumulativeReturns = ((1+StockReturns[name]).cumprod()-1)
        CumulativeReturns.plot(label=name)
    plt.legend()
    plt.show()
    
# 设置投资组合中股票的数目
numstocks = len(ticker_list)

# 平均分配每一项的权重
portfolio_weights_ew = np.repeat(1/numstocks, numstocks)

# 计算等权重组合的收益
StockReturns['Portfolio_EW'] = stock_return.mul(portfolio_weights_ew, axis=1).sum(axis=1)
#axis表示方向，axis=1表示从左往右

# 绘制累积收益曲线
cumulative_returns_plot(['Portfolio', 'Portfolio_EW'])

# 创建市值的数组
market_capitalizations = np.array([601.51, 469.25, 349.5, 310.48, 299.77, 356.94, 268.88, 331.57, 246.09])

# 计算市值权重
mcap_weights = market_capitalizations / np.sum(market_capitalizations)

# 计算市值加权的组合收益
StockReturns['Portfolio_MCap'] = stock_return.mul(mcap_weights, axis=1).sum(axis=1)
cumulative_returns_plot(['Portfolio', 'Portfolio_EW', 'Portfolio_MCap'])

# 计算相关矩阵 用于估算多只股票收益之间的线性关系
correlation_matrix = stock_return.corr()

# 输出相关矩阵
# print(correlation_matrix)

# 导入seaborn
import seaborn as sns

# 创建热图
sns.heatmap(correlation_matrix,
            annot=True,
            cmap="YlGnBu", 
            linewidths=0.3,
            annot_kws={"size": 8})

plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()


# 计算协方差矩阵
cov_mat = stock_return.cov()

# 年化协方差矩阵
cov_mat_annual = cov_mat * 252

# 输出协方差矩阵
# print(cov_mat_annual)

# 计算投资组合的标准差 
portfolio_volatility = np.sqrt(np.dot(portfolio_weights.T, np.dot(cov_mat_annual, portfolio_weights)))
# print(portfolio_volatility)

# 设置模拟的次数
number = 10000
# 设置空的numpy数组，用于存储每次模拟得到的权重、收益率和标准差
random_p = np.empty((number, 11))
# 设置随机数种子，这里是为了结果可重复
np.random.seed(123)

# 循环模拟10000次随机的投资组合
for i in range(number):
    # 生成9个随机数，并归一化，得到一组随机的权重数据
    random9 = np.random.random(9)
    random_weight = random9 / np.sum(random9)
    
    # 计算年化平均收益率
    mean_return = stock_return.mul(random_weight, axis=1).sum(axis=1).mean()
    annual_return = (1 + mean_return)**252 - 1 #一年有252个工作日

    # 计算年化的标准差，也称为波动率
    random_volatility = np.sqrt(np.dot(random_weight.T, np.dot(cov_mat_annual, random_weight)))

    # 将上面生成的权重，和计算得到的收益率、标准差存入数组random_p中
    random_p[i][:9] = random_weight
    random_p[i][9] = annual_return
    random_p[i][10] = random_volatility
    
# 将numpy数组转化成DataFrame数据框
RandomPortfolios = pd.DataFrame(random_p)
# 设置数据框RandomPortfolios每一列的名称
RandomPortfolios.columns = [ticker + "_weight" for ticker in ticker_list] + ['Returns', 'Volatility']
                         
# 绘制散点图
RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
plt.show()

# 找到标准差最小数据的索引值
min_index = RandomPortfolios.Volatility.idxmin()

# 在收益-风险散点图中突出风险最小的点
RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
x = RandomPortfolios.loc[min_index,'Volatility']
y = RandomPortfolios.loc[min_index,'Returns']
plt.scatter(x, y, color='red')   
plt.show()

# 提取最小波动组合对应的权重, 并转换成Numpy数组
GMV_weights = np.array(RandomPortfolios.iloc[min_index, 0:numstocks])
print(GMV_weights)

# 计算GMV投资组合收益
StockReturns['Portfolio_GMV'] = stock_return.mul(GMV_weights, axis=1).sum(axis=1)

# 绘制累积收益曲线
cumulative_returns_plot(['Portfolio_EW', 'Portfolio_MCap', 'Portfolio_GMV'])

# 设置无风险回报率为0
risk_free = 0


# 夏普比率代表投资人每多承担一分风险，可以拿到几分超额报酬
# 计算每项资产的夏普比率
RandomPortfolios['Sharpe'] = (RandomPortfolios.Returns - risk_free) / RandomPortfolios.Volatility

# 绘制收益-标准差的散点图，并用颜色描绘夏普比率
plt.scatter(RandomPortfolios.Volatility, RandomPortfolios.Returns, c=RandomPortfolios.Sharpe)
plt.colorbar(label='Sharpe Ratio')
plt.show()

# 找到夏普比率最大数据对应的索引值
max_index = RandomPortfolios.Sharpe.idxmax()

# 在收益-风险散点图中突出夏普比率最大的点
RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
x = RandomPortfolios.loc[max_index,'Volatility']
y = RandomPortfolios.loc[max_index,'Returns']
plt.scatter(x, y, color='red')   
plt.show()

# 提取最大夏普比率组合对应的权重，并转化为numpy数组
MSR_weights = np.array(RandomPortfolios.iloc[max_index, 0:numstocks])
print(MSR_weights)

# 计算MSR组合的收益
StockReturns['Portfolio_MSR'] = stock_return.mul(MSR_weights, axis=1).sum(axis=1)

# 绘制累积收益曲线
cumulative_returns_plot(['Portfolio_EW', 'Portfolio_MCap', 'Portfolio_GMV', 'Portfolio_MSR'])
    
    