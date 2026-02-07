import os
import streamlit as st
import backtrader as bt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import configparser
# import yfinance as yf  # 用于获取基础数据
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
from wordcloud import WordCloud
import jieba

# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_squared_error

from streamlit_echarts import st_echarts

import strategy.sma_strategy as strgsma
import strategy.top_buy_strategy as strgtop
import strategy.sma_double_strategy as strgsmadb
import strategy.grid_trading_strategy as strggrid

# matplotlib.use('Agg')
matplotlib.use('Agg')  # 使用非交互式后端'Agg'

def set_config():
    # 强制设置Streamlit版本兼容（需确保安装1.23.1）
    st.set_page_config(
        page_title="股票量化交易回测系统",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )


# 初始化会话状态
def init_session_state():
    if 'backtest_logs' not in st.session_state:
        st.session_state.backtest_logs = []
    if 'selected_stocks' not in st.session_state:
        st.session_state.selected_stocks = []
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None

stock_name = ""


def get_stock_name(stockcode: str):
    stockes = pd.read_csv('./data/stock_file_industry.csv', encoding="utf-8")
    if not stockes.empty:
        global stock_name
        # q = f"code == '{stockcode}'";
        # st.write(q)
        stock_name = stockes[stockes["code"] == stockcode]['code_name'].iloc[0]
        # st.write(stock_name)

    return stock_name


selected_value = None


def get_stock_selected(searchterm: str):
    # 这里实现你的搜索逻辑
    hs300s = pd.read_csv('./data/stock_file_hs300.csv', encoding="utf-8")
    # st.write("selected_value:"+selected_value)
    return hs300s["code"]


def show_words():
    # 定义文本
    text = "Colorful word clouds are amazing. They help to represent data visually and beautifully."
    text = "有色金属材料,黑色金属冶炼,电力热力生产，计算机通信和其他电子设备制造,有色金属矿采选业"
    text = st.text_area("原始新闻", "有色金属材料,黑色金属冶炼,电力热力生产，计算机通信和其他电子设备制造,有色金属矿采选业")
    stbtn_words = st.button("📥 生成词云", type="primary")

    if stbtn_words:
        if len(text) < 10:
            st.warning("请输入文字，不少于10个字符。")
            return
        st.write("股票分类词云生成分析")
        jall = jieba.lcut(text)
        print("---精准模式---")
        print(jall)
        res_1 = jieba._lcut_for_search(text)
        print("---搜索引擎模式---")
        words = jieba.lcut(text)
        # jieba.cut("这是一个测试文本", cut_all=False)
        jtext = " ".join(jall)
        st.write("股票分类词云生成")
        stopwords = set(["的", "你", "我", "道", "又", "他"])
        # 创建词云对象，设置颜色映射
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            font_path='./data/长仿宋体.ttf',
            stopwords=stopwords,
            colormap='plasma'  # 使用 'plasma' 颜色映射
        ).generate(jtext)
        # st.write("股票分类词云生成")
        # 显示词云
        # 用来正常显示中文
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        # 用来正常显示负号
        plt.rcParams["axes.unicode_minus"] = False
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        image_path_words = './data/words.png'
        plt.savefig(image_path_words)
        # plt.show()
        st.image(image_path_words, caption='股票分类词云')

def stock_k_priview(stockcode: str, stockname: str, data, display_data: int):
    if not data.empty:
        st.session_state.stock_data = data
        st.success(f"✅ 成功获取 {stockcode} 数据 ({len(data)} 条)")

        # 数据预览
        if display_data == 1:
            st.subheader("📋 数据预览")
            st.dataframe(data, use_container_width=True, column_config={'date': 'date-交易日期',
                                                                    'code': 'code-证券代码',
                                                                    'open': 'open-开盘价',
                                                                    'high': 'high-最高价',
                                                                    'low': 'low-最低价',
                                                                    'close': 'close-收盘价',
                                                                    'preclose': 'preclose-前收盘价',
                                                                    'volume': 'volume-成交量/股',
                                                                    'amount': 'amount-成交额/元',
                                                                    'adjustflag': 'adjustflag-复权状态',
                                                                    'turn': 'turn-换手率',
                                                                    'tradestatus': 'tradestatus-交易状态',
                                                                    'pctChg': 'pctChg-涨跌幅',
                                                                    'isST': 'isST-是否ST股',
                                                                    'peTTM': 'peTTM-滚动市盈率',
                                                                    'psTTM': 'psTTM-滚动市销率',
                                                                    'pcfNcfTTM': 'pcfNcfTTM-滚动市现率',
                                                                    'pbMRQ': 'pbMRQ-市净率'})

        # 价格走势图表
        st.subheader("📈 价格走势")
        fig = px.line(data, x="date", y=["high", "close"], title=f"收盘价走势 【{stockcode} - {stockname}】")
        st.plotly_chart(fig, use_container_width=True)


def stock_k_describe(stockcode: str, data):
    if not data.empty:
        # 数据统计信息
        st.subheader("📊 数据统计")
        # stats_col1, stats_col2 = st.columns(2)
        # with stats_col1:
        st.write("**基本统计** 【" + stockcode + "】")
        st.write(data.describe().rename(columns={'date': 'date-交易日期',
                                                 'code': 'code-证券代码',
                                                 'open': 'open-开盘价',
                                                 'high': 'high-最高价',
                                                 'low': 'low-最低价',
                                                 'close': 'close-收盘价',
                                                 'preclose': 'preclose-前收盘价',
                                                 'volume': 'volume-成交量/股',
                                                 'amount': 'amount-成交额/元',
                                                 'adjustflag': 'adjustflag-复权状态',
                                                 'turn': 'turn-换手率',
                                                 'tradestatus': 'tradestatus-交易状态',
                                                 'pctChg': 'pctChg-涨跌幅',
                                                 'isST': 'isST-是否ST股',
                                                 'peTTM': 'peTTM-滚动市盈率',
                                                 'psTTM': 'psTTM-滚动市销率',
                                                 'pcfNcfTTM': 'pcfNcfTTM-滚动市现率',
                                                 'pbMRQ': 'pbMRQ-市净率'}).round(2))
        # with stats_col2:
        #     st.write("**数据信息**")
        #     st.write(f"开始日期: {data['date'][0]}")
        #     st.write(f"结束日期: {data['date'].iloc[-1]}")
        #     st.write(f"最高价: {data['high'].max():.2f}")
        #     st.write(f"最低价: {data['low'].min():.2f}")
        #     st.write(f"平均收盘价: {data['close'].mean():.2f}")
    else:
        st.error("❌ 未获取到数据，请检查股票代码")


def stock_k_stat(data):
    st.subheader("📋 股价数据柱状图")
    st.write("**数据信息**")
    st.dataframe(data, use_container_width=True)
    # st.write(f"开始日期: {data['begin'][0]}")
    # st.write(f"结束日期: {data['end'][0]}")
    # st.write(f"最高价: {data['high'].max():.2f}")
    # st.write(f"最低价: {data['low'].min():.2f}")
    # st.write(f"平均收盘价: {data['mean'].max():.2f}")
    for index, stock in data.iterrows():
        st.markdown("---")
        # st.write(f"股票代码: {stock['code']}")
        # st.write(f"开始日期: {stock['begin']}")
        # st.write(f"结束日期: {stock['end']}")
        # st.write(f"最高价: {stock['high']:.2f}")
        # st.write(f"最低价: {stock['low']:.2f}")
        # st.write(f"平均收盘价: {stock['mean']:.2f}")
        # st.write(f"股票收盘价: {stock.close}")
        x, y1, y2, y3, y4 = stock.open, stock.close, stock.high, stock.low, stock.low
        # plt.figure(dpi=600)
        # 设置中文字体，如黑体或微软雅黑
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 设置图形大小
        plt.rcParams['figure.figsize'] = (9, 5)
        # 设置清晰度
        plt.rcParams['figure.dpi'] = 300
        wth = 0.5
        cats = ["open", "close", "high", "low", "mean"]
        values = np.array([x, y1, y2, y3, y4])
        vals = [x, y1, y2, y3, y4]
        plt.bar(cats, values, width=wth, label='price', color='red', edgecolor="black")
        plt.xlabel("价格类型")
        plt.ylabel("价格")
        plt.title("股票【"+stock['code']+"】价格柱状图")

        # plt.legend()
        # plt.show()
        # plt.savefig('./images/5-6'+str(index)+'.png')

        # 定义ECharts的配置
        option = {
            "title": {"text": "股票【"+stock['code']+"】价格柱状图"},
            "tooltip": {},
            "xAxis": {
                "data": cats
            },
            "yAxis": {},
            "series": [
                {
                    "name": "股价",
                    "type": "bar",
                    "data": vals
                }
            ]
        }

        # 在Streamlit应用中展示ECharts图表
        st_echarts(options=option, key="stock_price_"+stock['code'])

    # plt.savefig('images/5-6.png')

    # st.bar_chart(
    #     data,
    #     x="code",
    #     y=["high", "low", "mean"],
    # )

# 计算技术指标
def calculate_technical_indicators(df):
    """
    计算常用技术指标
    """
    ma_periods = [5, 10, 20, 60]
    rsi_period = 10
    # 计算均线
    for period in ma_periods:
        df[f'MA{period}'] = df['close'].rolling(window=period).mean()

    # 计算RSI指标
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 计算MACD指标
    # 计算12日和26日指数移动平均线
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()

    # 计算DIF和DEA
    df['DIF'] = df['EMA12'] - df['EMA26']
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()

    # 计算MACD柱状图
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])

    return df



# 绘制K线图和技术指标
def plot_stock_chart(df, stock_code):
    ma_periods = [5, 10, 20, 60]
    rsi_period = 10
    """
    使用plotly绘制股票K线图和技术指标
    """
    # 确定需要多少个子图
    show_rsi = True
    show_macd = True
    show_volume = True
    rows = 1
    if show_rsi:
        rows += 1
    if show_macd:
        rows += 1
    if show_volume:
        rows += 1

    # 创建子图
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "【"+stock_code+"】 K线图",
            "RSI指标" if show_rsi else None,
            "MACD指标" if show_macd else None,
            "成交量" if show_volume else None
        )
    )

    # 1. 添加K线图
    fig.add_trace(
        go.Candlestick(
            x=df.date,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="K线"
        ),
        row=1, col=1
    )

    # 添加均线
    show_ma = True
    if show_ma:
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
        for i, period in enumerate(ma_periods):
            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=df.date,
                    y=df[f'MA{period}'],
                    name=f"MA{period}",
                    line=dict(color=color, width=1)
                ),
                row=1, col=1
            )

    # 2. 添加RSI
    current_row = 2
    if show_rsi:
        fig.add_trace(
            go.Scatter(
                x=df.date,
                y=df['RSI'],
                name="RSI",
                line=dict(color='brown', width=1.5)
            ),
            row=current_row, col=1
        )
        # 添加超买超卖线
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=current_row, col=1)
        current_row += 1

    # 3. 添加MACD
    if show_macd:
        fig.add_trace(
            go.Scatter(
                x=df.date,
                y=df['DIF'],
                name="DIF",
                line=dict(color='blue', width=1)
            ),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.date,
                y=df['DEA'],
                name="DEA",
                line=dict(color='red', width=1)
            ),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Bar(
                x=df.date,
                y=df['MACD'],
                name="MACD",
                marker_color=df['MACD'].apply(lambda x: 'red' if x > 0 else 'green')
            ),
            row=current_row, col=1
        )
        current_row += 1

    # 4. 添加成交量
    if show_volume:
        # 根据涨跌设置成交量颜色
        colors = df['close'].diff().apply(lambda x: 'green' if x >= 0 else 'red')
        fig.add_trace(
            go.Bar(
                x=df.date,
                y=df['volume'],
                name="成交量",
                marker_color=colors
            ),
            row=current_row, col=1
        )

    # 更新布局
    fig.update_layout(
        height=600 + (150 * (rows - 1)),
        width=1200,
        title_x=0.5,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 更新X轴
    fig.update_xaxes(
        type='category',
        tickformat='%Y-%m-%d',
        tickangle=45,
        showspikes=True,
        spikemode='across'
    )

    return fig



# 显示数据表格
def show_data_table(df, stock_code):
    """
    显示股票数据表格
    """
    st.subheader("股票【"+stock_code+"】数据")
    show_ma = True
    show_rsi = True
    show_macd = True
    ma_periods = [5, 10, 20, 60]
    # 选择要显示的列
    columns_to_show = ['date', 'open', 'close', 'high', 'low', 'volume', 'pct_change']

    # 添加选中的均线列
    if show_ma:
        for period in ma_periods:
            columns_to_show.append(f'MA{period}')

    # 添加RSI列
    if show_rsi:
        columns_to_show.append('RSI')

    # 添加MACD相关列
    if show_macd:
        columns_to_show.extend(['DIF', 'DEA', 'MACD'])

    # 显示数据
    st.dataframe(
        df[columns_to_show].tail(50),
        width=1200,
        height=400,
        use_container_width=True
    )




def stock_data():
    # 选择股票数据
    selected_stocks = st.session_state.selected_stocks
    if len(selected_stocks) == 0:
        st.write("没有已选股票，请先执行选股。")
        return
    st.subheader("📋 选出来的股票")
    st.dataframe(selected_stocks, use_container_width=True)
    codes = selected_stocks["code"]
    # ["sz.300919", "sz.300759", "sh.600010", "sh.600018"]
    col1, col2, col3 = st.columns(3)
    with col1:
        # stock_code = st.text_input("股票代码", value="sh.600018", help="例如: sh.600018(上港集团)")
        options = st.multiselect(
            "查看股票信息",
            codes,
            default=[],
        )
        # st.write("You selected:", options)
    with col2:
        # start_date = st.date_input("开始日期", value=datetime(2022, 1, 1))
        # global selected_value
        # selected_value = st_searchbox(search_function, placeholder="输入搜索内容")
        # options = st.multiselect(
        #     "查看股票信息",
        #     ["sz.300919", "sz.300759", "sh.600010", "sh.600018"],
        #     default=[],
        # )
        st.write("You selected:", options)
    with col3:
        # end_date = st.date_input("结束日期", value=datetime(2024, 1, 1))
        stbtn_stock_data = st.button("📥 查看股票数据", type="primary")

    if stbtn_stock_data:
        if len(options) == 0:
            st.warning('请选择选股策略。', icon="⚠️")
            return
        # 根据股票代码获取股票名称
        with st.spinner("正在读取股票数据..."):
            try:
                # 使用yfinance获取数据
                # data = yf.download(stock_code, start=start_date, end=end_date)

                for skcd in options:
                    # dfstat = pd.DataFrame(columns=['date', 'code', 'begin', 'end', 'high', 'low', 'mean', 'close'])
                    stkname = get_stock_name(skcd)
                    data = pd.read_csv('./data/day_k_data' + skcd + '.csv', encoding="utf-8", parse_dates=True)
                    data['pct_change'] = round((data['close'] - data['open']) / data['open'], 4) * 100
                    # data['date'] = pd.to_datetime(data['date']).dt.date
                    stock_k_priview(skcd, stkname, data, 1)
                    # 计算技术指标
                    stock_data_with_indicators = calculate_technical_indicators(data.copy())
                    stock_k_describe(skcd, data)
                    # dfstat = dfstat.append({'code': skcd, 'begin': data['date'][0], 'end': data['date'].iloc[-1],
                    #                         'high': data['high'].max(), 'low': data['low'].min(),
                    #                         'close': data['close'].iloc[-1],
                    #                         'mean': data['close'].mean()}, ignore_index=True)
                    stock_k_stat(pd.DataFrame(columns=['date', 'open', 'code', 'begin', 'end', 'high', 'low', 'mean', 'close'], data={'open': data['open'].iloc[0], 'code': skcd, 'begin': data['date'][0], 'end': data['date'].iloc[-1],
                                            'high': data['high'].max(), 'low': data['low'].min(),
                                            'close': data['close'].iloc[-1],
                                            'mean': data['close'].mean()}, index=['row1']))
                    # dfstat = pd.DataFrame(columns=dfstat.columns)
                    # 绘制图表
                    fig = plot_stock_chart(stock_data_with_indicators, skcd)
                    st.plotly_chart(fig, use_container_width=True)

                    # 显示数据表格
                    show_data_table(stock_data_with_indicators, skcd)
                    start_date_str = data['date'][0]
                    end_date_str = data['date'].iloc[-1]

                    # 下载数据
                    csv = stock_data_with_indicators.to_csv().encode('utf-8')
                    st.download_button(
                        label="下载数据 (CSV)",
                        data=csv,
                        file_name=f"{skcd}_{start_date_str}_{end_date_str}.csv",
                        mime="text/csv",
                        help="点击下载当前股票数据的CSV文件"
                    )
            except Exception as e:
                st.error(f"❌ 获取数据失败: {str(e)}")


def choose_stock():
    st.subheader("⚡ 执行选股")
    stratege_select = st.selectbox(
        "策略选择",
        ["请选择选股策略", "基本面选股"]
    )
    st.write("你选择的是："+stratege_select)
    config_section = "BASE"
    if (stratege_select == "请选择选股策略"):
        st.warning('请选择选股策略。', icon="⚠️")
        return
    if(stratege_select == "基本面选股"):
        config_section = "BASE"
    config = configparser.ConfigParser()
    config.read("./config/choice_stock.ini", encoding='utf-8')
    # st.write(config.sections())
    base = config[config_section]

    # base["begin_date"]
    opt_industry = base.get("opt_industry")
    if len(opt_industry) == 0:
        st.warning("⚠️ 请先在「获取基础数据」页面加载股票数据")
    else:
        stbg = st.button("🔍 开始选股", type="primary")
        # 选股条件
        # st.markdown("### 选股条件")
        min_price = 10.0
        with st.expander("📜 查看选股条件"):
            for section in config.sections():
                st.write(section)
                for key, value in config.items(section):
                    st.write(f"{key} = {value}")
        # col1, col2 = st.columns(2)
        # with col1:
        #     min_price = st.number_input("最低价格", min_value=0.0, value=10.0)
        # with col2:
        #     max_price = st.number_input("最高价格", min_value=0.0, value=100.0)
        if stbg:
            with st.spinner("正在执行选股..."):
                # 简单选股逻辑示例
                # data = st.session_state.stock_data
                stock_all = pd.read_csv('./data/total_kday_data_2024.csv', encoding="gbk")
                date_tg = base.get("begin_date")
                stcok_date = stock_all.query("date == @date_tg")
                st.dataframe(stcok_date, use_container_width=True)
                price_bg = float(base.get("price_bg"))
                price_ed = float(base.get("price_ed"))
                volume = int(base.get("volume"))
                turn = float(base.get("turn"))
                opt_industry = base.get("opt_industry")
                ln = len(opt_industry)
                if len(opt_industry) > 0:
                    opt_industry = opt_industry[1:ln-1]
                    opt_industry = opt_industry.replace("'", "")
                    # st.write(opt_industry)
                    opt_indx = np.array(opt_industry.split(", "))
                # st.write(opt_indx)
                # & volume > @volume & turn >= @turn   & industry in @opt_industry
                # "industry.str.contains('" + opt_industry + "')"['B09有色金属矿采选业', 'C31黑色金属冶炼和压延加工业', 'D44电力、热力生产和供应业']
                selected = stcok_date.query("close >= @price_bg & close <= @price_ed & close > preclose & volume > @volume & industry in @opt_indx")
                # stock_qs = stock_all.query("industry == @opt_industry")
                # 展示选股结果
                st.subheader("📋 选股结果")
                st.dataframe(selected, use_container_width=True)
                st.session_state.selected_stocks = selected
                st.success(f"✅ 选股完成，共筛选出 {len(selected)} 个符合条件的股票")
                # st.dataframe(selected[['close', 'volume']], use_container_width=True)

                # 可视化选股结果
                fig = px.scatter(
                    selected,
                    x=selected.index,
                    y='close',
                    size='volume',
                    title="选股结果价格分布",
                    labels={'Close': '收盘价', 'Volume': '成交量'}
                )
                st.plotly_chart(fig, use_container_width=True)


def stock_prediction():
    # st.subheader("📈 股票趋势预测")
    selected_stocks = st.session_state.selected_stocks
    if len(selected_stocks) == 0:
        st.write("没有已选股票，请先执行选股。")
        return
    st.subheader("📋 选出来的股票")
    st.dataframe(selected_stocks, use_container_width=True)
    codes = selected_stocks["code"]
    # ["sz.300919", "sz.300759", "sh.600010", "sh.600018"]
    col1, col2, col3 = st.columns(3)
    with col1:
        # stock_code = st.text_input("股票代码", value="sh.600018", help="例如: sh.600018(上港集团)")
        options = st.multiselect(
            "选择要预测的股票",
            codes,
            default=[],
        )
        # st.write("You selected:", options)
    with col2:
        # start_date = st.date_input("开始日期", value=datetime(2022, 1, 1))
        # global selected_value
        # selected_value = st_searchbox(search_function, placeholder="输入搜索内容")
        # options = st.multiselect(
        #     "查看股票信息",
        #     ["sz.300919", "sz.300759", "sh.600010", "sh.600018"],
        #     default=[],
        # )
        st.write("You selected:", options)
    with col3:
        # end_date = st.date_input("结束日期", value=datetime(2024, 1, 1))
        stbtn_stock_data = st.button("📥 执行趋势预测", type="primary")

    if stbtn_stock_data:
        if len(options) == 0:
            st.warning('请选择选股策略。', icon="⚠️")
            return
        # 根据股票代码获取股票名称
        with st.spinner("读取股票数据"):
            # st.write("正在读取股票数据...")
            for skcd in options:
                data = pd.read_csv('./data/day_k_data' + skcd + '.csv', encoding="utf-8", parse_dates=True)
                data['pct_change'] = round((data['close'] - data['open']) / data['open'], 4) * 100
                # data['date'] = pd.to_datetime(data['date']).dt.date
                stkname = get_stock_name(skcd)
                stock_k_priview(skcd, stkname, data, 0)

                # df['date'] = pd.to_datetime(df['date'])
                # df.set_index('date', inplace=True)
                # df = df.apply(pd.to_numeric)

                # 采用ARIMA(5, 1, 2)
                # 模型进行时间序列预测，包含以下关键步骤：
                #
                # 数据集划分（80 % 训练 / 20 % 测试）
                # 模型参数拟合
                # 预测结果评估（MSE指标）
                #  arima_res = arima_predict(data.close, 30)

                # 通过Matplotlib实现多维度数据可视化，包含：
                #
                # 历史价格趋势
                # 训练集 / 测试集划分
                # 预测结果对比
                # 未来价格预测
                #  plot_results(data, arima_res.train, arima_res.test, arima_res.forecast, arima_res.future)

            # plt.figure(figsize=(16, 6))
            # plt.title('历史收盘价', fontsize=20)
            # plt.plot(df_stock['close'])
            # plt.xlabel('日期', fontsize=18)
            # plt.ylabel('收盘价 RMB', fontsize=18)
            # # plt.show()
            # image_path_stock = './data/selected_stocks.png'
            # plt.savefig(image_path_stock)
            # st.image(image_path_stock, caption='股票分类词云')


def plot_results(full_data, train, test, forecast, future, title="ARIMA预测结果"):
    """可视化预测结果"""
    plt.figure(figsize=(14, 7))
    plt.plot(full_data.index, full_data, label='实际价格', color='blue', alpha=0.5)
    plt.plot(train.index, train, label='训练集', color='green')
    plt.plot(test.index, test, label='测试集', color='orange')
    plt.plot(forecast.index, forecast, label='测试集预测', color='red', linestyle='--')
    plt.plot(future.index, future, label='未来预测', color='purple', linestyle='-.')

    plt.title(title, fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('价格', fontsize=12)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def arima_predict(series, steps=30):
    """ARIMA模型预测"""
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    # 拟合ARIMA(1, 1, 1)模型
    # model = ARIMA(df.close, order=(1, 1, 1))
    model = ARIMA(train, order=(5, 1, 2))
    model_fit = model.fit()
    print(model_fit.summary())

    forecast = model_fit.forecast(steps=len(test))
    mse = mean_squared_error(test, forecast)
    print(f'测试集MSE: {mse:.4f}')

    future_forecast = model_fit.forecast(steps=steps)
    return {
        'train': train,
        'test': test,
        'forecast': forecast,
        'future': future_forecast
    }


def choose_stock_strategy():
    st.subheader("🎯 选股策略配置")
    # "热点行业策略",
    strategy_type = st.selectbox(
        "选择选股策略",
        ["基本面选股"]
    )
    # stock_file_industry.csv

    st.markdown("### 策略参数设置")
    st.info("📝 策略逻辑: 热点行业+均线向上+最低市值+换手率+滚动市盈率+市净率")
    if strategy_type != "热点行业策略1":
        st.markdown("#### 行业策略")
        dfids = pd.read_csv('./data/stock_file_industry.csv', encoding="utf-8")
        dfids = dfids.dropna(axis=0)
        dfin = dfids["industry"].drop_duplicates(keep="first", inplace=False)
        # .distinct()
        # st.write(dfin.columns)
        # st.write(dfin)
        col_fw, col_bg, col_ed = st.columns(3)
        with col_fw:
            stock_scope = st.selectbox(
                "股票范围",
                ["沪深300", "上证50"]
            )
        with col_bg:
            begin_date = st.date_input("指定日期", value=datetime(2024, 1, 1))
        with col_ed:
            end_date = ''
            # end_date = st.date_input("结束日期", value=datetime(2024, 12, 31))

        # ["C39计算机、通信和其他电子设备制造业", "M73研究和试验发展", "C31黑色金属冶炼和压延加工业", "G55水上运输业"]dfids[dfids["industry"]].distinct(),
        # col_industry = st.columns(1)
        # with col_industry:
        opt_industry = st.multiselect(
                "证券行业分类",
                dfin,
                default=[],
        )
        # opt_industry = st.selectbox(
        #     "证券行业分类",
        #     dfin,
        # )
        # with col2:
            # short_ma = st.number_input("日线均值SMA", min_value=5, max_value=60, value=5)
            # peTTM = st.text_input("滚动市盈率", value="8.8", help="最低8.8")
        # with col3:
            # market_value = st.number_input("公司市值", min_value=10, max_value=200, value=20)
            # pbMRQ = st.text_input("市净率", value="0.9", help="最低0.9")

    if strategy_type != "基本面选股1":
        st.markdown("#### 基本面策略")
        col1, col2, col3 = st.columns(3)
        with col1:
            # market_value = st.number_input("公司市值", min_value=10, max_value=200, value=20)
            market_value = st.slider("公司市值", 50, 500, 100)
            pb_ratio = st.slider("市净率(PB)最大值", 0, 20, 5)
            turn = st.slider("换手率turn", 0.1, 5.0, 0.15)
            price_bg = st.number_input("收盘价最低股价", value=10, help="最低10")
        with col2:
            peTTM = st.slider("滚动市盈率peTTM", 10, 100, 30)
            pe_ratio = st.slider("市盈率(PE)最大值", 10, 100, 30)
            volume = st.slider("成交量volume(股)", 10000000, 80000000, 20000000)
            price_ed = st.number_input("收盘价最高股价", value=50, help="最高50")
            # st.text_input("成交量(股)", value="32000", help="最低8.8")
        with col3:
            pc = st.slider("滚动市现率(PC)最小值", 0, 100, 10)
            roe = st.slider("净资产收益率(ROE)最小值", 0, 100, 10)
            pbMRQ = st.slider("市净率pbMRQ", 1.5, 100.0, 2.0)

    # 保存策略
    if st.button("💾 保存选股策略", type="primary"):
        if len(opt_industry) == 0:
            # st.write("请选择证券行业分类。")
            st.warning('保存失败，请选择证券行业分类。', icon="⚠️")
            # st.toast("请选择证券行业分类。", icon="😍")
            return
        date_bg = datetime.strptime(str(begin_date), "%Y-%m-%d")
        # st.write("====================="+str(begin_date))
        yr = str(date_bg.year)
        edt = yr+"-12-31"
        file_exists = os.path.exists('./data/trade_date_'+yr+'.csv')
        if file_exists:
            st.write("")
        else:
            st.warning("没有找到"+yr+"年的交易时间文件。")
            return
        df_date = pd.read_csv('./data/trade_date_'+yr+'.csv', encoding="utf-8", parse_dates=True)
        bgdt = str(begin_date)
        # st.write(bgdt)
        dfselect = df_date.query("calendar_date == @bgdt")
        # st.write(len(dfselect))
        # st.write(dfselect["is_trading_day"].iloc[0])
        is_td = int(dfselect["is_trading_day"].iloc[0])
        if len(dfselect) != 1:
            st.warning("你选择的时间【"+str(begin_date)+"】不是交易时间，请重新选择。")
            return
        elif is_td == 0:
            st.warning("你选择的时间【" + str(begin_date) + "】不是交易时间，请重新选择。")
            return

        st.write(
            f"选股开始日期：{begin_date},结束日期：{edt},证监会行业分类:{opt_industry}，股票范围：{stock_scope},公司市值: {market_value},滚动市盈率:{peTTM},市净率:{pbMRQ},最低股价:{price_bg},最高股价:{price_ed}")

        config = configparser.ConfigParser()
        if not config.has_section("INDUSTRY"):
            config.add_section("INDUSTRY")
        config.set("INDUSTRY", "opt_industry", str(opt_industry))
        config.set("INDUSTRY", "stock_scope", str(stock_scope))
        if not config.has_section("BASE"):
            config.add_section("BASE")
        config.set("BASE", "opt_industry", str(opt_industry))
        config.set("BASE", "stock_scope", str(stock_scope))
        config.set("BASE", "begin_date", str(begin_date))
        config.set("BASE", "end_date", str(end_date))
        config.set("BASE", "begin_date", str(begin_date))
        config.set("BASE", "pe_ratio", str(pe_ratio))
        config.set("BASE", "pb_ratio", str(pb_ratio))
        config.set("BASE", "volume", str(volume))
        config.set("BASE", "pc", str(pc))
        config.set("BASE", "roe", str(roe))
        config.set("BASE", "price_bg", str(price_bg))
        config.set("BASE", "price_ed", str(price_ed))
        config.set("BASE", "turn", str(turn))
        config.set("BASE", "market_value", str(market_value))
        config.set("BASE", "peTTM", str(peTTM))
        config.set("BASE", "pbMRQ", str(pbMRQ))

        with open("./config/choice_stock.ini", 'w', encoding='utf-8') as configfile:
            config.write(configfile)
        st.success("✅ 选股策略保存成功")
        config.read("./config/choice_stock.ini", encoding='utf-8')
        base = config["BASE"]
        # st.write(base["begin_date"])
        st.write(base.get("begin_date"))

from numpy.random import default_rng as rng

def back_test_result():
    if st.session_state.backtest_results is None:
        st.warning("⚠️ 请先执行交易回测")
    else:
        results_array = st.session_state.backtest_results
        # st.write(results_array)
        stocks = []
        incomes = []
        # rel_ana_df = pd.DataFrame()
        st.session_state.exec_type = 'g'
        # 关键指标卡片
        for results in results_array:
            # df_rel = pd.DataFrame(results)
            # rel_ana_df = pd.concat([rel_ana_df, df_rel[["initial_cash", "final_value"]]], axis=0)
            # stkname = get_stock_name(results['stock_code'])
            st.subheader("📊 股票回测结果展示")
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                # 返回的是投资组合的‌总资产价值‌，包括可用现金（cash）和所有持仓资产（如股票、期货等）的当前市场价值。这个值反映了账户的总净资产。
                st.metric("最终总资产", f"¥{results['final_value']:,.2f}")
                # 仅返回账户中‌可用的现金余额‌，不包含任何持仓资产的价值。
                st.metric("‌可用现金", f"¥{results['valid_cash']:,.2f}")
            with col2:
                st.metric("总收益", f"¥{results['total_return']:,.2f}")
                st.metric("总收益率",
                          f"¥{((results['final_value'] - results['initial_cash']) / results['initial_cash']) * 100:,.2f}%")
            with col3:
                st.metric("夏普比率： ", f"{results['sharpe']:,.2f}")
                st.metric("交易次数", f"{results['trade_num']:,.2f}")
            with col4:
                st.metric("最大回撤金额", f"{results['moneydown']:,.2f}")
                st.metric("最大回撤比率", f"{results['drawdown']:,.2f}%")

            # 收益对比图表
            # st.markdown("### 📈 收益走势分析")
            # 模拟收益数据
            # dates = pd.date_range(start="2024-01-02", end="2024-12-31", periods=1)
            # equity_curve = np.linspace(results['initial_cash'], results['final_value'], 50)
            # benchmark = np.linspace(results['initial_cash'],
            #                         results['initial_cash'] * (1 + results['total_return'] / 200), 50)
            #
            # plot_data = pd.DataFrame({
            #     "日期": dates,
            #     "策略收益": equity_curve,
            #     "基准收益": benchmark
            # })
            #
            # fig = px.line(
            #     plot_data,
            #     x="日期",
            #     y=["策略收益", "基准收益"],
            #     title="策略收益 vs 基准收益",
            #     labels={"value": "资产价值", "variable": "收益类型"}
            # )
            # st.plotly_chart(fig, use_container_width=True)

            # 风险指标分析
            # st.markdown("### 🚨 风险指标分析")
            # col1, col2 = st.columns(2)
            # with col1:
            # 参考 https://zhuanlan.zhihu.com/p/526634713
            st.write("**最大回撤分析**")
            drawdown_data = pd.DataFrame({
                    "回撤幅度(%)": [results['drawdown'], 30, 50, 20, 15],
                    "对比基准": ["策略最大回撤", "行业平均", "市场平均", "风险阈值", "警戒线"]
            })
            fig_dd = px.bar(drawdown_data, x="对比基准", y="回撤幅度(%)", title="最大回撤对比")
            st.plotly_chart(fig_dd, use_container_width=True)

            # with col2:
            #     st.write("**月度收益分布**")
            #     monthly_returns = np.random.normal(results['total_return'] / 24, 2, 24)  # 模拟月度收益
            #     month_data = pd.DataFrame({
            #         "月份": [f"{i + 1}月" for i in range(24)],
            #         "收益率(%)": monthly_returns
            #     })
            #     fig_mr = px.bar(month_data, x="月份", y="收益率(%)", title="月度收益率分布")
            #     fig_mr.update_traces(marker_color=np.where(monthly_returns >= 0, 'green', 'red'))
            #     st.plotly_chart(fig_mr, use_container_width=True)

            # 回测报告下载
            st.markdown("### 📄 回测报告")
            report_text = f"""
            # 量化交易回测报告
            ## 回测概览
            - 初始资金: ¥{results['initial_cash']:,.2f}
            - 最终资产: ¥{results['final_value']:,.2f}
            - 总收益: ¥{results['pnl']:,.2f}
            - 总收益率: {results['total_return']:.2f}%
            - 夏普比率: {results['sharpe']:.3f}
            - 最大回撤: {results['drawdown']:.2f}%
    
            ## 交易统计
            - 总交易次数: {results['trade_num']}
            - 平均每笔收益: {results['pnl'] / (len(st.session_state.backtest_logs) // 2) if st.session_state.backtest_logs else 0:.2f}
            """
            st.download_button(
                label="📥 下载回测报告",
                data=report_text,
                file_name="回测报告.md",
                mime="text/markdown"
            )
            # st.write(pd.DataFrame(incomes).T)
            # st.write("stocks")
            # st.write(stocks)
            # st.write(rng(0).standard_normal((20, 3)))

            # st.markdown("### 收益投入对比分析")
            # chart_data = pd.DataFrame(
            #     pd.DataFrame(incomes).T,
            #     columns=stocks)
            #
            # st.line_chart(chart_data, use_container_width=True)
            # st.markdown("---")


def  back_testing_strategy():
    st.subheader("📋 回测策略管理")

    # 选择回测策略 "RSI超买超卖策略",
    backtest_strategy = st.selectbox(
        "选择回测策略",
        ["简单均线策略", "双均线交叉策略", "打板策略", "网格交易策略"]
    )

    # 策略参数配置
    ma_period = 5
    ma_5 = 5
    ma_10 = 10
    grid_gap = 5
    grid_size = 1000
    grid_floor = 50
    grid_top = 500
    st.subheader("📋 策略参数配置")
    # if backtest_strategy == "简单均线策略":
    st.markdown("***简单均线策略***")
    ma_period = st.slider("均线周期", 5, 60, 5)
    st.session_state.ma_period = ma_period
    # elif backtest_strategy == "双均线交叉策略":
    st.markdown("***双均线交叉策略***")
    col1, col2 = st.columns(2)
    with col1:
        ma_5 = st.slider("短均线周期", 5, 20, 5)
    with col2:
        ma_10 = st.slider("长均线周期", 10, 60, 10)
    # elif backtest_strategy == "网格交易策略":
    st.markdown("***网格交易策略***")
    col_gap, col_size, col_floor, col_top = st.columns(4)
    with col_gap:
        grid_gap = st.slider("网格间距", 2, 200, 10)
    with col_size:
        grid_size = st.slider("单次交易数量", 100, 20000, 1000)
    with col_floor:
        grid_floor = st.slider("网格下限", 10, 200, 20)
    with col_top:
        grid_top = st.slider("网格上限", 10, 500, 1000)

    # 佣金和滑点设置
    st.markdown("### 交易成本设置")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        commission = st.number_input("佣金比例(‰)", min_value=0.0, max_value=10.0, value=0.1) / 1000
    with col2:
        slippage = st.number_input("滑点(‰)", min_value=0.0, max_value=10.0, value=0.1) / 1000
    with col3:
        initial_cash = st.number_input("初始资金", min_value=1000, value=100000)
    with col4:
        per_type = st.selectbox(
            "每次交易方式",
            ["股数", "比例"]
        )
    with col5:
        per_size = st.number_input("每次交易股数或资金比例", min_value=20, value=1000)

    # 保存回测策略
    if st.button("💾 保存回测策略", type="primary"):
        if per_type == "比例" and per_size > 100:
            st.warning("保存失败，选择比例时不能超过【100】.")
            return;
        st.success("✅ 回测策略保存成功")
        # 保存策略参数到会话状态
        st.session_state.backtest_params = {
            "strategy": backtest_strategy,
            "ma_period": ma_period,
            "ma_5": ma_5,
            "ma_10": ma_10,
            "grid_gap": grid_gap,
            "grid_size": grid_size,
            "grid_floor": grid_floor,
            "grid_top": grid_top,
            "commission": commission,
            "slippage": slippage,
            "pertype": per_type,
            "persize": per_size,
            "initial_cash": initial_cash
        }

def stock_trading():
    # 检查数据和策略
    if st.session_state.stock_data is None:
        st.warning("⚠️ 请先在「获取基础数据」页面加载股票数据")
    elif "backtest_params" not in st.session_state:
        st.warning("⚠️ 请先在「交易回测策略管理」页面配置并保存策略")
    else:
        # 回测参数确认
        params = st.session_state.backtest_params
        # st.markdown("### 回测参数确认,如果选出的股票大于三支只跑前面三只")
        st.markdown("### 选择的策略")
        st.info(f"{params['strategy']}")
        st.markdown("### 简单均线策略策略")
        st.info(f"均线周期：{params['ma_period']}")
        st.markdown("### 双均线交叉策略")
        st.info(f"短均线周期：{params['ma_5']}，长均线周期：{params['ma_10']}")
        st.markdown("### 网格交易策略")
        st.info(f"网格间距：{params['grid_gap']}，单次交易数量：{params['grid_size']}，网格下限：{params['grid_floor']}，网格上限：{params['grid_top']}，")
        st.markdown("### 交易成本设置")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.info(f"佣金比例: {params['commission'] * 1000:.1f}‰")
        with col2:
            st.info(f"滑点: {params['slippage'] * 1000:.1f}‰")
            st.write(params['slippage'])
        with col3:
            st.info(f"初始资金: ¥{params['initial_cash']:,}")
        with col4:
            st.info(f"购买方式: {params['pertype']}")
        with col5:
            st.info(f"每次购买数量: {params['persize']:.1f} 股数 or %")

        st.subheader("▶️ 执行交易回测")

        selected_stocks = st.session_state.selected_stocks
        codes = selected_stocks["code"]
        col01, col02, col03 = st.columns(3)
        with col01:
            options = st.multiselect(
                "选择要回测的股票",
                codes,
                default=[],
            )
        with col02:
            st.write("You selected:", options)
        with col03:
            stbtn_testing = st.button("🚀 执行单股回测", type="primary")
            stbtn_testgroup = st.button("🚀 执行组合回测", type="primary")

        if stbtn_testgroup:
            with st.spinner("正在执行组合回测，请稍候..."):
                # 重置回测日志
                st.session_state.backtest_logs = []
                st.session_state.backtest_results = []
                st.session_state.exec_type = 'g'
                st.session_state.backtest_logs.append("开始执行组合股票回测。代码："+("".join(options)))
                backtrading_testing_group(options, strategy_name=params['strategy'], params=params)


                # 回测日志
                with st.expander("📜 查看回测日志"):
                    st.write("回测日志")
                    for log in st.session_state.backtest_logs:
                        st.write(log)

        if stbtn_testing:
            with st.spinner("正在执行单股回测，请稍候..."):
                # 重置回测日志
                st.session_state.backtest_logs = []
                st.session_state.backtest_results = []
                st.session_state.exec_type = 's'
                for stok_cod in options:
                    st.session_state.backtest_logs.append("开始执行股票回测。代码：【"+stok_cod+"】")
                    # simple_trading(stock_code=stok_cod)
                    backtrading_testing_single(stock_code=stok_cod, strategy_name=params['strategy'], params=params)


                # 回测日志
                with st.expander("📜 查看回测日志"):
                    st.write("回测日志")
                    for log in st.session_state.backtest_logs:
                        st.write(log)


def simple_trading(stock_code: str):
    plt.rcParams['figure.dpi'] = 300
    # 设置中文字体，如黑体或微软雅黑
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置图形大小
    plt.rcParams['figure.figsize'] = (9, 6)
    data = bt.feeds.GenericCSVData(dataname='./data/day_k_data'+stock_code+'.csv',
                                   nullvalue=0.0,
                                   dtformat=('%Y-%m-%d'),
                                   tmformat=('%H:%M:%S'),
                                   datetime=0,
                                   open=2,
                                   high=3,
                                   low=4,
                                   close=5,
                                   volume=7,
                                   timeframe=bt.TimeFrame.Minutes,
                                   compression=10)
    cerebro = bt.Cerebro()
    cerebro.adddata(data)

    # 为输出的图形设置标题
    # plt.title("finance data show")
    cerebro.run()
    print("save image")
    image_path_stock = './sp_' + stock_code + '.png'
    fig = cerebro.plot(iplot=False, show=False)
    # print(cerebro.broker.getvalue())
    # plt.plot([1, 2, 3], [4, 5, 6])
    fig[0][0].savefig(image_path_stock)
    st.image(image_path_stock, caption='股票'+stock_code+'回测结果');
    # plt.show()


def backtrading_testing_single(stock_code: str, strategy_name: str, params):
    params = st.session_state.backtest_params
    selected_stocks = st.session_state.selected_stocks

    # 创建主控制器
    cerebro1 = bt.Cerebro()
    # 获取数据
    # df = pd.read_csv('./data/day_k_data'+stock_code+'.csv', encoding="utf-8", parse_dates=True,
    #                  index_col='date')
    # # df.index = pd.to_datetime(df.date)
    # df = df[['open', 'high', 'low', 'close', 'volume']]
    # # 将数据加载至回测系统
    # data = bt.feeds.PandasData(dataname=df)
    data = bt.feeds.GenericCSVData(dataname='./data/day_k_data' + stock_code + '.csv',
                                   nullvalue=0.0,
                                   dtformat=('%Y-%m-%d'),
                                   tmformat=('%H:%M:%S'),
                                   datetime=0,
                                   open=2,
                                   high=3,
                                   low=4,
                                   close=5,
                                   volume=7,
                                   timeframe=bt.TimeFrame.Minutes,
                                   compression=10)
    cerebro1.adddata(data)
    # ["简单均线策略", "双均线交叉策略", "打板策略", "网格交易策略"]
    if strategy_name == "简单均线策略":
        idx = cerebro1.addstrategy(strgsma.SmaStrategy, trade_base=params)
    if strategy_name == "双均线交叉策略":
        # bt.strategies.SMA_CrossOver trade_base
        idx = cerebro1.addstrategy(strgsmadb.SmaDoubleStrategy, trade_base=params)
    if strategy_name == "打板策略":
        idx = cerebro1.addstrategy(strgtop.TopBuyStrategy, trade_base=params)
    if strategy_name == "网格交易策略":
        idx = cerebro1.addstrategy(strggrid.GridTradingStrategy, trade_base=params)
    # 设置默认所有策略都用的买入设置，策略，数量percents
    if params['pertype'] == "比例":
        if strategy_name == "网格交易策略":
            cerebro1.addsizer(bt.sizers.FixedSize, stake=int(params["grid_size"]))
        else:
            cerebro1.addsizer(bt.sizers.PercentSizer, percents=int(params["persize"]))
    else:
        if strategy_name == "网格交易策略":
            cerebro1.addsizer(bt.sizers.FixedSize, stake=int(params["grid_size"]))
        else:
            cerebro1.addsizer(bt.sizers.FixedSize, stake=int(params["persize"]))

    # specify size to a strategy
    # cerebro.addsizer_byidx(idx, bt.sizers.SizerFix, stake=qts)
    # 可以同时执行多个策略
    # cerebro.addstrategy(SmaStrategy,maperiod=12)
    # broker设置资金、手续费
    cerebro1.broker.setcash(float(params["initial_cash"]))
    cerebro1.broker.setcommission(commission=float(params["commission"]))
    # 设置滑点（模拟市场冲击）
    cerebro1.broker.set_slippage_perc(float(params['slippage']))
    # 百分比滑点配置示例
    # cerebro1.broker.set_slippage_perc(
    #     slip_perc=float(params['slippage']),  # 0.1%滑点
    #     slip_open=True,  # 开盘价也应用滑点
    #     slip_match=True,  # 匹配价格时考虑滑点
    #     slip_out=False  # 不允许超出bar价格范围
    # )
    st.write('执行回测股票：'+stock_code)
    # st.write('期初总资金: %.2f' % cerebro1.broker.getvalue())
    # 计算最大回撤相关指标
    cerebro1.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown')
    # 回撤期间
    cerebro1.addanalyzer(bt.analyzers.TimeDrawDown, _name='_TimeDrawDown')
    # 计算年化夏普比率
    cerebro1.addanalyzer(bt.analyzers.SharpeRatio, _name='_SharpeRatio', timeframe=bt.TimeFrame.Days, annualize=True,
                        riskfreerate=0)
    # 交易统计信息，如获胜、失败次数
    cerebro1.addanalyzer(bt.analyzers.TradeAnalyzer, _name='_TradeAnalyzer')
    # 收益率
    cerebro1.addanalyzer(bt.analyzers.Returns, _name='_Returns')
    # 收益期间
    cerebro1.addanalyzer(bt.analyzers.TimeReturn, _name='_TimeReturn')
    result = cerebro1.run(maxcpus=1)
    strat = result[0]
    # st.write('期末总资金: %.2f' % cerebro1.broker.getvalue())

    image_path_stock = './sp_' + stock_code + '.png'
    fig = cerebro1.plot(iplot=False, show=False)
    fig[0][0].savefig(image_path_stock)
    st.image(image_path_stock, caption='股票' + stock_code + '回测结果')
    # cerebro1.plot()
    # print(cerebro1.datas)
    # print(dir(cerebro1))

    # 提取结果
    # st.metric('最终资金: %.2f' % cerebro1.broker.getvalue())
    # st.metric("收益率： ", result[0].analyzers._Returns.get_analysis()['rtot'])
    # st.metric("--------------- 收益期间 -----------------")
    # st.metric(result[0].analyzers._TimeReturn.get_analysis())
    # st.metric("--------------- 最大回撤相关指标 -----------------")
    # st.metric(result[0].analyzers._DrawDown.get_analysis())
    # st.metric("--------------- 回撤期间 -----------------")
    # st.metric(result[0].analyzers._TimeDrawDown.get_analysis())
    # st.metric("夏普比率： ", result[0].analyzers._SharpeRatio.get_analysis()['sharperatio'])

    # st.metric("最终资金:", f"¥{cerebro1.broker.getvalue():,.2f}")
    # st.metric("收益率： ", f"{result[0].analyzers._Returns.get_analysis()['rtot']:,.2f}")

    # st.metric("--------------- 收益期间 -----------------")
    # st.metric("收益期间", result[0].analyzers._TimeReturn.get_analysis())
    # st.metric("--------------- 最大回撤相关指标 -----------------")
    # st.metric("最大回撤相关指标", result[0].analyzers._DrawDown.get_analysis())
    # st.metric("--------------- 回撤期间 -----------------")
    # st.metric("回撤期间", result[0].analyzers._TimeDrawDown.get_analysis())

    # st.metric("夏普比率： ", f"{result[0].analyzers._SharpeRatio.get_analysis()['sharperatio']:,.2f}")

    # 显示回测结果摘要
    st.success("✅ 回测执行完成！")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # 返回的是投资组合的‌总资产价值‌，包括可用现金（cash）和所有持仓资产（如股票、期货等）的当前市场价值。这个值反映了账户的总净资产。
        st.metric("最终总资产", f"¥{cerebro1.broker.getvalue():,.2f}")
        # 仅返回账户中‌可用的现金余额‌，不包含任何持仓资产的价值。
        st.metric("‌可用现金", f"¥{cerebro1.broker.getcash():,.2f}")
    with col2:
        st.metric("总收益", f"¥{cerebro1.broker.getvalue() - params['initial_cash']:,.2f}")
        st.metric("总收益率", f"¥{((cerebro1.broker.getvalue() - params['initial_cash']) / params['initial_cash']) * 100:,.2f}%")
    with col3:
        sharp = strat.analyzers._SharpeRatio.get_analysis()
        if sharp['sharperatio']:
            st.metric("夏普比率： ", f"{round(sharp['sharperatio'],2):,.2f}")
        ta = strat.analyzers._TradeAnalyzer.get_analysis()
        # st.write(ta)
        trade_num = 0;
        if ta.total.total != 0:
            trade_num = ta.total.closed if hasattr(ta.total, 'closed') else sum(v for v in ta.values() if hasattr(v, 'closed'))
            st.metric("交易次数", trade_num)
    with col4:
        dd = strat.analyzers._DrawDown.get_analysis()
        # print('Max Drawdown: %.2f%%' % dd['max']['drawdown'])'OrderedDict([('sharperatio', 0.7246096598590239)])'
        if dd:
            st.metric("最大回撤金额", round(dd['max']['moneydown'], 2))
            st.metric("最大回撤比率", f"{round(dd['max']['drawdown'], 2):,.2f}%")

        # 保存回测结果
        bk_rel = {
            "stock_code": stock_code,
            "final_value": cerebro1.broker.getvalue(),
            "valid_cash": cerebro1.broker.getcash(),
            "initial_cash": params['initial_cash'],
            "pnl": cerebro1.broker.getvalue() - params['initial_cash'],
            "sharpe": strat.analyzers._SharpeRatio.get_analysis()['sharperatio'],
            "moneydown": strat.analyzers._DrawDown.get_analysis()['max']['moneydown'],
            "drawdown": strat.analyzers._DrawDown.get_analysis()['max']['drawdown'],
            "return": strat.analyzers._Returns.get_analysis(),
            "total_return": cerebro1.broker.getvalue() - params['initial_cash'],
            "trade_num": trade_num,
            # strat.analyzers.returns.get_analysis()['rtot'] * 100
        }
        st.session_state.backtest_results.append(bk_rel)



    # st.write(dd)
    # st.write(ta)
    # st.write('Total Trades:', ta.total.closed if hasattr(ta.total, 'closed') else sum(
    #     v for v in ta.values() if hasattr(v, 'closed')))


def backtrading_testing_group(options, strategy_name: str, params):
    params = st.session_state.backtest_params
    selected_stocks = st.session_state.selected_stocks

    # 创建主控制器
    cerebro1 = bt.Cerebro()
    for stock_code in options:
        data = bt.feeds.GenericCSVData(dataname='./data/day_k_data' + stock_code + '.csv',
                                   nullvalue=0.0,
                                   dtformat=('%Y-%m-%d'),
                                   tmformat=('%H:%M:%S'),
                                   datetime=0,
                                   open=2,
                                   high=3,
                                   low=4,
                                   close=5,
                                   volume=7,
                                   timeframe=bt.TimeFrame.Minutes,
                                   compression=10)
        cerebro1.adddata(data, name=stock_code)
    # ["简单均线策略", "双均线交叉策略", "打板策略", "网格交易策略"] df.fillna(axis=0, method="ffill")
    if strategy_name == "简单均线策略":
        idx = cerebro1.addstrategy(strgsma.SmaStrategy, trade_base=params)
    if strategy_name == "双均线交叉策略":
        # bt.strategies.SMA_CrossOver trade_base
        idx = cerebro1.addstrategy(strgsmadb.SmaDoubleStrategy, trade_base=params)
    if strategy_name == "打板策略":
        idx = cerebro1.addstrategy(strgtop.TopBuyStrategy, trade_base=params)
    if strategy_name == "网格交易策略":
        idx = cerebro1.addstrategy(strggrid.GridTradingStrategy, trade_base=params)
    # 设置默认所有策略都用的买入设置，策略，数量percents
    if params['pertype'] == "比例":
        if strategy_name == "网格交易策略":
            cerebro1.addsizer(bt.sizers.FixedSize, stake=int(params["grid_size"]))
        else:
            cerebro1.addsizer(bt.sizers.PercentSizer, percents=int(params["persize"]))
    else:
        if strategy_name == "网格交易策略":
            cerebro1.addsizer(bt.sizers.FixedSize, stake=int(params["grid_size"]))
        else:
            cerebro1.addsizer(bt.sizers.FixedSize, stake=int(params["persize"]))
    # specify size to a strategy
    # cerebro.addsizer_byidx(idx, bt.sizers.SizerFix, percents=qts)
    # 可以同时执行多个策略
    # cerebro.addstrategy(SmaStrategy,maperiod=12)
    # broker设置资金、手续费
    cerebro1.broker.setcash(float(params["initial_cash"]))
    cerebro1.broker.setcommission(commission=float(params["commission"]))
    # 设置滑点（模拟市场冲击）
    cerebro1.broker.set_slippage_perc(float(params['slippage']))
    st.write('执行回测股票：'+(" ".join(options)))
    # st.write('期初总资金: %.2f' % cerebro1.broker.getvalue())
    # 计算最大回撤相关指标
    cerebro1.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown')
    # 回撤期间
    cerebro1.addanalyzer(bt.analyzers.TimeDrawDown, _name='_TimeDrawDown')
    # 计算年化夏普比率
    cerebro1.addanalyzer(bt.analyzers.SharpeRatio, _name='_SharpeRatio', timeframe=bt.TimeFrame.Days, annualize=True,
                        riskfreerate=0)
    # 交易统计信息，如获胜、失败次数
    cerebro1.addanalyzer(bt.analyzers.TradeAnalyzer, _name='_TradeAnalyzer')
    # 收益率
    cerebro1.addanalyzer(bt.analyzers.Returns, _name='_Returns')
    # 收益期间
    cerebro1.addanalyzer(bt.analyzers.TimeReturn, _name='_TimeReturn')
    # 添加月度收益率分析器
    # cerebro1.addanalyzer(bt.analyzers.MonthlyReturn, _name='_MonthlyReturn')
    result = cerebro1.run(maxcpus=1)
    strat = result[0]
    # st.write('期末总资金: %.2f' % cerebro1.broker.getvalue())

    image_path_stock = './sp_' + stock_code + '.png'
    fig = cerebro1.plot(iplot=False, show=False)
    fig[0][0].savefig(image_path_stock)
    st.image(image_path_stock, caption='股票' + stock_code + '回测结果')
    # cerebro1.plot()
    # print(cerebro1.datas)
    # print(dir(cerebro1))

    # 提取结果
    # st.metric('最终资金: %.2f' % cerebro1.broker.getvalue())
    # st.metric("收益率： ", result[0].analyzers._Returns.get_analysis()['rtot'])
    # st.metric("--------------- 收益期间 -----------------")
    # st.metric(result[0].analyzers._TimeReturn.get_analysis())
    # st.metric("--------------- 最大回撤相关指标 -----------------")
    # st.metric(result[0].analyzers._DrawDown.get_analysis())
    # st.metric("--------------- 回撤期间 -----------------")
    # st.metric(result[0].analyzers._TimeDrawDown.get_analysis())
    # st.metric("夏普比率： ", result[0].analyzers._SharpeRatio.get_analysis()['sharperatio'])

    # st.metric("最终资金:", f"¥{cerebro1.broker.getvalue():,.2f}")
    # st.metric("收益率： ", f"{result[0].analyzers._Returns.get_analysis()['rtot']:,.2f}")

    # st.metric("--------------- 收益期间 -----------------")
    # st.metric("收益期间", result[0].analyzers._TimeReturn.get_analysis())
    # st.metric("--------------- 最大回撤相关指标 -----------------")
    # st.metric("最大回撤相关指标", result[0].analyzers._DrawDown.get_analysis())
    # st.metric("--------------- 回撤期间 -----------------")
    # st.metric("回撤期间", result[0].analyzers._TimeDrawDown.get_analysis())

    # st.metric("夏普比率： ", f"{result[0].analyzers._SharpeRatio.get_analysis()['sharperatio']:,.2f}")
    # st.write(strat.analyzers._Returns.get_analysis())
    # 显示回测结果摘要
    st.success("✅ 回测执行完成！")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # 返回的是投资组合的‌总资产价值‌，包括可用现金（cash）和所有持仓资产（如股票、期货等）的当前市场价值。这个值反映了账户的总净资产。
        st.metric("最终总资产", f"¥{cerebro1.broker.getvalue():,.2f}")
        # 仅返回账户中‌可用的现金余额‌，不包含任何持仓资产的价值。
        st.metric("‌可用现金", f"¥{cerebro1.broker.getcash():,.2f}")
    with col2:
        st.metric("总收益", f"¥{cerebro1.broker.getvalue() - params['initial_cash']:,.2f}")
        st.metric("总收益率", f"¥{((cerebro1.broker.getvalue() - params['initial_cash']) / params['initial_cash']) * 100:,.2f}%")
    with col3:
        sharp = strat.analyzers._SharpeRatio.get_analysis()
        if sharp['sharperatio']:
            st.metric("夏普比率： ", f"{round(sharp['sharperatio'],2):,.2f}")
        ta = strat.analyzers._TradeAnalyzer.get_analysis()
        # st.write(ta)
        trade_num = 0
        if ta.total.total != 0:
            trade_num = ta.total.closed if hasattr(ta.total, 'closed') else sum(v for v in ta.values() if hasattr(v, 'closed'))
            st.metric("交易次数", trade_num)
    with col4:
        dd = strat.analyzers._DrawDown.get_analysis()
        # print('Max Drawdown: %.2f%%' % dd['max']['drawdown'])'OrderedDict([('sharperatio', 0.7246096598590239)])'
        if dd:
            st.metric("最大回撤金额", round(dd['max']['moneydown'], 2))
            st.metric("最大回撤比率", f"{round(dd['max']['drawdown'], 2):,.2f}%")

    # 保存回测结果
    bk_rel1 = {
        "stock_code": stock_code,
        "final_value": cerebro1.broker.getvalue(),
        "initial_cash": params['initial_cash'],
        "valid_cash": cerebro1.broker.getcash(), #_MonthlyReturn
        "pnl": cerebro1.broker.getvalue() - params['initial_cash'],
        "sharpe": strat.analyzers._SharpeRatio.get_analysis()['sharperatio'],
        "return": strat.analyzers._Returns.get_analysis(),
        "moneydown": strat.analyzers._DrawDown.get_analysis()['max']['moneydown'],
        "drawdown": strat.analyzers._DrawDown.get_analysis()['max']['drawdown'],
        "total_return": cerebro1.broker.getvalue() - params['initial_cash'],
        "trade_num": trade_num,
    }
    st.session_state.backtest_results.append(bk_rel1)

    # st.write(dd)
    # st.write(ta)
    # st.write('Total Trades:', ta.total.closed if hasattr(ta.total, 'closed') else sum(
    #     v for v in ta.values() if hasattr(v, 'closed')))


def stock_app():
    # --------------------------
    # 侧边栏导航
    # --------------------------
    st.sidebar.title("📊 股票量化交易回测系统")
    st.sidebar.markdown("---")

    # 导航选项
    nav_options = {
        "分类词云": "🌥️ 股票分类事件词云",
        "选股策略": "🎯 选股策略管理",
        "执行选股": "⚡ 执行选股",
        "选股数据": "🚠 查看选股数据",
        "回测策略": "📋 交易回测策略管理",
        "趋势预测": "📈 股票趋势预测",
        "执行回测": "🚩 执行交易回测",
        "回测结果": "📊 回测结果展示"
    }

    selected_page = st.sidebar.radio(
        "导航菜单",
        list(nav_options.keys()),
        format_func=lambda x: nav_options[x]
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "📌 系统说明\n"
        "- 基于Streamlit 1.23.1开发\n"
        "- 集成Backtrader 1.9.78.123回测框架\n"
        "- 支持从baostock获取基础数据"
    )

    # --------------------------
    # 主页面内容
    # --------------------------
    st.title(nav_options[selected_page])
    st.markdown("---")

    if selected_page == "分类词云":
        show_words()

    # 1. 获取基础数据
    if selected_page == "选股数据":
        stock_data()

    # 2. 选股策略管理
    elif selected_page == "选股策略":
        choose_stock_strategy()

    # 3. 执行选股
    elif selected_page == "执行选股":
        choose_stock()

    # 4. 交易回测策略管理
    elif selected_page == "回测策略":
        back_testing_strategy()

    # 4.1 趋势预测
    elif selected_page == "趋势预测":
        stock_prediction()

    # 5. 执行交易回测
    elif selected_page == "执行回测":
        stock_trading()

    # 6. 回测结果展示
    elif selected_page == "回测结果":
        back_test_result()

    # 页脚
    st.markdown("---")
    st.caption("© 2025 股票量化交易回测系统 | 基于 Streamlit 1.23.1 和 Backtrader 1.9.78.123 开发")
    # 打板策略,当日收盘价涨停时买入（做多），当收盘价下跌超5%卖出（做空）


if __name__ == '__main__':
    set_config()
    init_session_state()
    stock_app()
