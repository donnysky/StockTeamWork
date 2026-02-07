# StockTeamWork
A Very Simple Python Program for Analysing China's Stock. Stock Data Analyse and Research.

## 一、环境安装

## **代码环境**

```python
windows 11
# 版本对应关系 https://blog.csdn.net/fxqrd19287/article/details/150493372
# 清华大学镜像网站下载最新版Anaconda：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/?C=M&O=D
Anaconda3-2021.05-Windows-x86_64.exe
# 直接升级
# conda install python=3.8
# 新安装的，直接安装这个 Anaconda3-2021.05对应python 3.8
```

## **依赖安装**

```python
运行cmd时，在图标上右键，以管理员运行，执行下面的安装
pip install streamlit==1.23.1 --user
# backtrader==1.9.78.123 plotly pandas numpy
pip install backtrader
pip install backtrader[plotting] -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install matplotlib
pip install matplotlib --upgrade --user
pip install baostock -i https://pypi.org/simple
pip install streamlit-searchbox --user
pip install streamlit-echarts  --user
pip install numpy==1.20.3
# module 'pandas' has no attribute 'Float64Index'
pip install pandas==1.5.3 --user
pip install numexpr==2.8.6 --user
# 版本兼容问题，pyfolio跑不起来
pip install pyfolio --user
pip install pyfolio-reloaded==0.9.3 --user
# 改用这个
pip install quantstats --upgrade --no-cache-dir
pip install quantstats==0.0.62
# conda里安装
pip install TA_Lib-0.4.28-cp38-cp38-win_amd64.whl
# 还要安装yfinance

# 这个不行得换一个
pip install streamlit-aggrid --user

pip install wordcloud
pip install jieba
pip install seaborn
pip install statsmodels

pip install --user --upgrade aws-sam-cli
pip install markupsafe==2.0.1
# pip install tensorflow==2.10.0 --user

获取数据失败: cannot import name 'AggFuncType' from 'pandas._typing' (c:\programdata\anaconda3\lib\site-packages\pandas_typing.py)
```

## git操作

```
git clone https://github.com/donnysky/StockTeamWork.git
git checkout main
git pull
git status
git add xxx.py
git commit -m '增加了下载基础数据工具类'
git push -u origin main
```



## 二、资源

```python
# emoji图标
https://www.emojiall.com/zh-hans/keywords/%E5%9B%BE%E8%A1%A8
# streamlit api reference， 注意右上角的版本，选择当前使用对应的版本，有些api旧版本没有
https://docs.streamlit.io/develop/api-reference/data/st.dataframe
```



## 三、主程序和运行
StockWorks.ipynb

```python
streamlit程序运行：在主目录运行，
# streamlit run backtesting.py
# D:\django\StockTeamWork
# D:\django>jupyter notebook
```



## 四、程序工具类
StockBackProgram.py

## 五、数据存放目录
data

## 六、处理步骤

```
首先使用 describe() 函数对数据集 data 进行描述性统计,然后通过 round(2) 函数将统计结果四舍五入到小数点后两位
```



## 七、相关技术概念

**数据集的快速统计概览**

```
data.describe()：
这是 pandas 库中 DataFrame 对象的一个方法，用于生成数据集的描述性统计摘要。它会自动计算每列数值数据的以下统计量：
coun：非缺失值的数量。
mean：平均值，反映数据的中心趋势。
std：标准差，衡量数据的离散程度（标准差越大，数据波动越剧烈）。
min：最小值，表示数据的下限。
25%(Q1)：第一四分位数（下四分位数），即数据中25%的值小于该值。
50%(Q2/中位数)：第二四分位数（中位数），数据的中间值。
75%（Q3)：第三四分位数（上四分位数），即数据中75%的值小于该值。
max：最大值，表示数据的上限。
这些统计量能快速了解数据的分布、中心位置和离散情况。
```

backtrader

![image-20251207215502098](C:\Users\done5\AppData\Roaming\Typora\typora-user-images\image-20251207215502098.png) 


## 八、股票模型训练和预测
```
cd predict
stock_train.ipynb:用于训练股票模型
stock_predict.ipynb：用于预测股票未来6天的收盘价
```