import streamlit as st
import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import configparser
# import yfinance as yf  # 用于获取基础数据
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib

from streamlit_echarts import st_echarts

import strategy.sma_strategy as strgsma
import strategy.buy_top_strategy as strgtop

matplotlib.use('Agg')

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


def stock_k_priview(stockcode: str, stockname: str, data):
    if not data.empty:
        st.session_state.stock_data = data
        st.success(f"✅ 成功获取 {stockcode} 数据 ({len(data)} 条)")

        # 数据预览
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
        st.write(f"股票代码: {stock['code']}")
        st.write(f"开始日期: {stock['begin']}")
        st.write(f"结束日期: {stock['end']}")
        st.write(f"最高价: {stock['high']:.2f}")
        st.write(f"最低价: {stock['low']:.2f}")
        st.write(f"平均收盘价: {stock['mean']:.2f}")
        st.write(f"股票收盘价: {stock.close}")
        x, y1, y2, y3 = stock.close, stock.high, stock.low, stock.low
        # plt.figure(dpi=600)
        # 设置中文字体，如黑体或微软雅黑
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 设置图形大小
        plt.rcParams['figure.figsize'] = (9, 5)
        # 设置清晰度
        plt.rcParams['figure.dpi'] = 300
        wth = 0.5
        cats = ["close", "high", "low", "mean"]
        values = np.array([x, y1, y2, y3])
        vals = [x, y1, y2, y3]
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


def stock_data():
    # 股票日线数据
    col1, col2, col3 = st.columns(3)
    with col1:
        stock_code = st.text_input("股票代码", value="sh.600018", help="例如: sh.600018(上港集团)")
    with col2:
        # start_date = st.date_input("开始日期", value=datetime(2022, 1, 1))
        # global selected_value
        # selected_value = st_searchbox(search_function, placeholder="输入搜索内容")
        options = st.multiselect(
            "查看股票信息",
            ["sz.300919", "sz.300759", "sh.600010", "sh.600018"],
            default=[],
        )

        st.write("You selected:", options)
    with col3:
        end_date = st.date_input("结束日期", value=datetime(2024, 1, 1))

    if st.button("📥 查看K线数据", type="primary"):
        # 根据股票代码获取股票名称
        with st.spinner("正在读取股票数据..."):
            try:
                # 使用yfinance获取数据
                # data = yf.download(stock_code, start=start_date, end=end_date)

                for skcd in options:
                    dfstat = pd.DataFrame(columns=['date', 'code', 'begin', 'end', 'high', 'low', 'mean', 'close'])
                    stkname = get_stock_name(skcd)
                    data = pd.read_csv('./data/day_k_data' + skcd + '.csv', encoding="utf-8")
                    stock_k_priview(skcd, stkname, data)
                    stock_k_describe(skcd, data)
                    dfstat = dfstat.append({'code': skcd, 'begin': data['date'][0], 'end': data['date'].iloc[-1],
                                            'high': data['high'].max(), 'low': data['low'].min(),
                                            'close': data['close'].iloc[-1],
                                            'mean': data['close'].mean()}, ignore_index=True)
                    stock_k_stat(pd.DataFrame(columns=['date', 'code', 'begin', 'end', 'high', 'low', 'mean', 'close'], data={'code': skcd, 'begin': data['date'][0], 'end': data['date'].iloc[-1],
                                            'high': data['high'].max(), 'low': data['low'].min(),
                                            'close': data['close'].iloc[-1],
                                            'mean': data['close'].mean()}, index=['row1']))
                    # dfstat = pd.DataFrame(columns=dfstat.columns)
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
        # 选股条件
        st.markdown("### 选股条件")
        min_price = 10.0
        for section in config.sections():
            st.write(section)
            for key, value in config.items(section):
                st.write(f"{key} = {value}")
        # col1, col2 = st.columns(2)
        # with col1:
        #     min_price = st.number_input("最低价格", min_value=0.0, value=10.0)
        # with col2:
        #     max_price = st.number_input("最高价格", min_value=0.0, value=100.0)
        if st.button("🔍 开始选股", type="primary"):
            with st.spinner("正在执行选股..."):
                # 简单选股逻辑示例
                # data = st.session_state.stock_data
                stock_all = pd.read_csv('./data/total_kday_data_2024.csv', encoding="gbk")
                stcok10 = stock_all.head(10)
                st.dataframe(stcok10, use_container_width=True)
                selected = stcok10[(stcok10['close'] >= min_price)]
                st.session_state.selected_stocks = selected
                st.success(f"✅ 选股完成，共筛选出 {len(selected)} 个交易日符合条件")

                # 展示选股结果
                st.subheader("📋 选股结果")
                st.dataframe(selected[['close', 'volume']], use_container_width=True)

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
    st.subheader("📈 股票趋势预测")



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
            begin_date = st.date_input("开始日期", value=datetime(2024, 1, 1))
        with col_ed:
            end_date = st.date_input("结束日期", value=datetime(2024, 12, 31))

        # ["C39计算机、通信和其他电子设备制造业", "M73研究和试验发展", "C31黑色金属冶炼和压延加工业", "G55水上运输业"]dfids[dfids["industry"]].distinct(),
        # col_industry = st.columns(1)
        # with col_industry:
        opt_industry = st.multiselect(
                "证券行业分类",
                dfin,
                default=[],
        )
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
            turn = st.slider("换手率", 1, 50, 10)
            price_bg = st.number_input("最低股价", value=10, help="最低10")
        with col2:
            peTTM = st.slider("滚动市盈率peTTM", 10, 100, 30)
            pe_ratio = st.slider("市盈率(PE)最大值", 10, 100, 30)
            volume = st.slider("成交量(股)", 10000000, 80000000, 20000000)
            price_ed = st.number_input("最高股价", value=50, help="最高50")
            # st.text_input("成交量(股)", value="32000", help="最低8.8")
        with col3:
            pc = st.slider("滚动市现率(PC)最小值", 0, 100, 10)
            roe = st.slider("净资产收益率(ROE)最小值", 0, 100, 10)
            pbMRQ = st.slider("pbMRQ", 1.5, 100.0, 2.0)

    # 保存策略
    if st.button("💾 保存选股策略", type="primary"):
        if len(opt_industry) == 0:
            # st.write("请选择证券行业分类。")
            st.warning('保存失败，请选择证券行业分类。', icon="⚠️")
            # st.toast("请选择证券行业分类。", icon="😍")
            return
        st.write(
            f"选股开始日期：{begin_date},开始日期：{end_date},证监会行业分类:{opt_industry}，股票范围：{stock_scope},公司市值: {market_value},滚动市盈率:{peTTM},市净率:{pbMRQ},最低股价:{price_bg},最高股价:{price_ed}")

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


def back_test_result():
    st.subheader("📊 回测结果展示")

    if st.session_state.backtest_results is None:
        st.warning("⚠️ 请先执行交易回测")
    else:
        results = st.session_state.backtest_results

        # 关键指标卡片
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("初始资金", f"¥{results['initial_cash']:,.2f}")
        with col2:
            st.metric("最终资产", f"¥{results['final_value']:,.2f}",
                      delta=f"{results['pnl']:,.2f}", delta_color="normal")
        with col3:
            st.metric("总收益率", f"{results['total_return']:.2f}%")
        with col4:
            st.metric("夏普比率", f"{results['sharpe']:.3f}")

        # 收益对比图表
        st.markdown("### 📈 收益走势分析")
        # 模拟收益数据
        dates = pd.date_range(start="2022-01-01", end="2024-01-01", periods=50)
        equity_curve = np.linspace(results['initial_cash'], results['final_value'], 50)
        benchmark = np.linspace(results['initial_cash'],
                                results['initial_cash'] * (1 + results['total_return'] / 200), 50)

        plot_data = pd.DataFrame({
            "日期": dates,
            "策略收益": equity_curve,
            "基准收益": benchmark
        })

        fig = px.line(
            plot_data,
            x="日期",
            y=["策略收益", "基准收益"],
            title="策略收益 vs 基准收益",
            labels={"value": "资产价值", "variable": "收益类型"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # 风险指标分析
        st.markdown("### 🚨 风险指标分析")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**最大回撤分析**")
            drawdown_data = pd.DataFrame({
                "回撤幅度(%)": [results['drawdown'], 5, 10, 15, 20],
                "对比基准": ["策略最大回撤", "行业平均", "市场平均", "风险阈值", "警戒线"]
            })
            fig_dd = px.bar(drawdown_data, x="对比基准", y="回撤幅度(%)", title="最大回撤对比")
            st.plotly_chart(fig_dd, use_container_width=True)

        with col2:
            st.write("**月度收益分布**")
            monthly_returns = np.random.normal(results['total_return'] / 24, 2, 24)  # 模拟月度收益
            month_data = pd.DataFrame({
                "月份": [f"{i + 1}月" for i in range(24)],
                "收益率(%)": monthly_returns
            })
            fig_mr = px.bar(month_data, x="月份", y="收益率(%)", title="月度收益率分布")
            fig_mr.update_traces(marker_color=np.where(monthly_returns >= 0, 'green', 'red'))
            st.plotly_chart(fig_mr, use_container_width=True)

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

        ## 风险分析
        - 策略最大回撤: {results['drawdown']:.2f}%
        - 收益波动率: {np.std(monthly_returns):.2f}%
        - 胜率: {len([x for x in monthly_returns if x > 0]) / len(monthly_returns) * 100:.1f}%

        ## 交易统计
        - 总交易次数: {len(st.session_state.backtest_logs) // 2 if st.session_state.backtest_logs else 0}
        - 平均每笔收益: {results['pnl'] / (len(st.session_state.backtest_logs) // 2) if st.session_state.backtest_logs else 0:.2f}
        """
        st.download_button(
            label="📥 下载回测报告",
            data=report_text,
            file_name="回测报告.md",
            mime="text/markdown"
        )


def  back_testing_strategy():
    st.subheader("📋 回测策略管理")

    # 选择回测策略
    backtest_strategy = st.selectbox(
        "选择回测策略",
        ["简单均线策略", "双均线交叉策略", "RSI超买超卖策略", "自定义策略"]
    )

    # 策略参数配置
    st.markdown("### 策略参数配置")
    if backtest_strategy == "简单均线策略":
        ma_period = st.slider("均线周期", 5, 100, 15)
        st.session_state.ma_period = ma_period

    elif backtest_strategy == "双均线交叉策略":
        col1, col2 = st.columns(2)
        with col1:
            fast_ma = st.slider("快速均线周期", 5, 50, 10)
        with col2:
            slow_ma = st.slider("慢速均线周期", 10, 200, 60)

    # 佣金和滑点设置
    st.markdown("### 交易成本设置")
    col1, col2, col3 = st.columns(3)
    with col1:
        commission = st.number_input("佣金比例(‰)", min_value=0.0, max_value=10.0, value=0.5) / 1000
    with col2:
        slippage = st.number_input("滑点(‰)", min_value=0.0, max_value=10.0, value=0.1) / 1000
    with col3:
        initial_cash = st.number_input("初始资金", min_value=1000, value=100000)

    # 保存回测策略
    if st.button("💾 保存回测策略", type="primary"):
        st.success("✅ 回测策略保存成功")
        # 保存策略参数到会话状态
        st.session_state.backtest_params = {
            "strategy": backtest_strategy,
            "commission": commission,
            "slippage": slippage,
            "initial_cash": initial_cash
        }

def stock_trading():
    st.subheader("▶️ 执行交易回测")

    # 检查数据和策略
    if st.session_state.stock_data is None:
        st.warning("⚠️ 请先在「获取基础数据」页面加载股票数据")
    elif "backtest_params" not in st.session_state:
        st.warning("⚠️ 请先在「交易回测策略管理」页面配置并保存策略")
    else:
        # 回测参数确认
        params = st.session_state.backtest_params
        st.markdown("### 回测参数确认")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"初始资金: ¥{params['initial_cash']:,}")
        with col2:
            st.info(f"佣金比例: {params['commission'] * 1000:.1f}‰")
        with col3:
            st.info(f"滑点: {params['slippage'] * 1000:.1f}‰")

        if st.button("🚀 开始回测", type="primary"):
            with st.spinner("正在执行回测，请稍候..."):
                # 重置回测日志
                st.session_state.backtest_logs = []

                # # 准备数据
                # data = st.session_state.stock_data
                # cerebro = bt.Cerebro()
                #
                # # 添加数据
                # feed = bt.feeds.PandasData(dataname=data)
                # cerebro.adddata(feed)
                #
                # # 设置初始资金
                # cerebro.broker.setcash(params['initial_cash'])
                #
                # # 设置佣金和滑点
                # cerebro.broker.setcommission(commission=params['commission'])
                # cerebro.broker.set_slippage_fixed(params['slippage'])
                #
                # # 添加策略
                # cerebro.addstrategy(SimpleStrategy, maperiod=st.session_state.get('ma_period', 15))
                #
                # # 添加分析器
                # cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
                # cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
                # cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

                # 运行回测
                # results = cerebro.run()
                # strat = results[0]

                # 创建主控制器
                cerebro1 = bt.Cerebro()
                # 导入策略参数寻优 range(3, 31)
                # cerebro.optstrategy(Sma5Strategy,maperiod=5)
                # 获取数据
                df = pd.read_csv('./data/day_k_datash.600000.csv', encoding="utf-8", parse_dates=True,
                                 index_col='date')
                # df.index = pd.to_datetime(df.date)
                df = df[['open', 'high', 'low', 'close', 'volume']]
                # 将数据加载至回测系统
                data = bt.feeds.PandasData(dataname=df)
                cerebro1.adddata(data)
                idx = cerebro1.addstrategy(strgtop.UpperStrategy, maperiod=12)
                # 设置默认所有策略都用的买入设置，策略，数量
                cerebro1.addsizer(bt.sizers.FixedSize, stake=10000)
                # specify size to a strategy
                # cerebro.addsizer_byidx(idx, bt.sizers.SizerFix, stake=qts)
                # 可以同时执行多个策略
                # cerebro.addstrategy(SmaStrategy,maperiod=12)
                # broker设置资金、手续费
                cerebro1.broker.setcash(100000)
                cerebro1.broker.setcommission(commission=0.0001)
                print('期初总资金: %.2f' %
                      cerebro1.broker.getvalue())
                results = cerebro1.run(maxcpus=1)
                strat = results[0]
                print('期末总资金: %.2f' % cerebro1.broker.getvalue())
                cerebro1.plot()
                # cerebro1.plot()
                # print(cerebro1.datas)
                # print(dir(cerebro1))

                # 保存回测结果
                st.session_state.backtest_results = {
                    "final_value": cerebro1.broker.getvalue(),
                    "initial_cash": params['initial_cash'],
                    "pnl": cerebro1.broker.getvalue() - params['initial_cash'],
                    # "sharpe": strat.analyzers.sharpe.get_analysis().get('sharperatio', 0),
                    # "drawdown": strat.analyzers.drawdown.get_analysis()['max']['drawdown'],
                    # "total_return": strat.analyzers.returns.get_analysis()['rtot'] * 100
                }

                # 显示回测结果摘要
                st.success("✅ 回测执行完成！")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("最终资产", f"¥{cerebro1.broker.getvalue():,.2f}")
                with col2:
                    st.metric("总收益", f"¥{cerebro1.broker.getvalue() - params['initial_cash']:,.2f}")
                with col3:
                    st.metric("总收益率", "需计算")
                with col4:
                    st.metric("最大回撤", "需计算")

                # 回测日志
                with st.expander("📜 查看回测日志", expanded=False):
                    for log in st.session_state.backtest_logs:
                        st.write(log)

def stock_app():
    # --------------------------
    # 侧边栏导航
    # --------------------------
    st.sidebar.title("📊 股票量化交易回测系统")
    st.sidebar.markdown("---")

    # 导航选项
    nav_options = {
        "选股策略": "🎯 选股策略管理",
        "执行选股": "⚡ 执行选股",
        "选股数据": "📈 查看选股数据",
        "回测策略": "📋 交易回测策略管理",
        "趋势预测": "📈 股票趋势预测",
        "执行回测": "▶️ 执行交易回测",
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
