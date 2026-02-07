
import numpy as np
# 导入backtrader平台
import backtrader as bt
import streamlit as st

# --------------------------
# 全局配置和工具函数
# --------------------------
class GridTradingStrategy(bt.Strategy):
    """基础回测策略模板"""
    params = (
        ('maperiod', 15),
    )

    def __init__(self, trade_base):
        # 设置网格交易下限
        self.buy_price = trade_base["grid_floor"]
        # 设置网格交易上限
        self.sell_price = trade_base["grid_top"]
        self.grid_floor = trade_base["grid_floor"]
        self.grid_top = trade_base["grid_top"]
        # 网格间距
        self.grid_gap = trade_base["grid_gap"]
        # 网格上限
        self.trade_size = trade_base["grid_size"]
        self.buyprice = None
        self.order = None
        self.buycomm = None
        # 买入价格记录
        self.price = []
        # 交易获利记录
        self.profit = []
        # 单词清仓获利
        self.close_profit = 0
        self.smas = {d._name: bt.indicators.SMA(d.close, period=trade_base["ma_period"]) for d in self.datas}

    def start(self):
        self.log(f"策略回测开始，起始资金：{self.broker.startingcash}。")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                # self.price.append(order.executed.price)
                self.log(
                    f'买入执行, 价格: {order.executed.price:.2f}, 数量：{order.executed.size}, 成本: {order.executed.value:.2f}, 佣金: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                # self.log(f'notify_order \n买入价格：{order.executed.price}，\
                #                          数量：{order.executed.size}，\
                #                          金额：{order.executed.value}，\
                #                          平均持仓成本：{self.getposition(self.data).price}')
            else:
                self.log(
                    f'卖出执行, 价格: {order.executed.price:.2f}, 数量：{order.executed.size}， 成本: {order.executed.value:.2f}, 佣金: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                # profit = round(order.executed.price - self.price[-1], 1)
                # self.close_profit += profit * self.trade_size
                # self.profit.append(profit * abs(order.executed.size))
                # self.log(f'notify_order \n卖出价格: {order.executed.price},\
                #                          数量：{order.executed.size}，\
                #                          本次交易获利：{self.profit[-1]:.1f}，\
                #                          剩余股票数量:{self.position.size}')
                # self.price.pop()
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/被拒绝')

        self.order = None

    def notify_trade(self, trade):
        # if not trade.isclosed:
        #     return
        if trade.size == 0:
            self.log(f'notify_trade 股票已清仓，共获利：{self.close_profit}')
            # print("----------------------------------------------------")

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        st.session_state.backtest_logs.append(f'{dt.isoformat()} - {txt}')

    def stop(self):
        self.log(f'执行stop函数！\n策略回测已结束，累计交易次数：{len(self.profit)}，\
                累计盈利:{np.sum(self.profit)}，\n交易单次盈利记录：{self.profit}')

    def next(self):
        # 空仓情况下，只能执行买入操作
        for dat in self.datas:
            position = self.getposition(dat)
            sma = self.smas[dat._name]
            if not position:
                # 初始化买入网格下限价格
                self.close_profit = 0

                # 执行买入操作的条件
                # 条件1：当前收盘价＞网格下限
                buy_con_1 = dat.close[0] > self.buy_price
                # 条件2：当前收盘价＜网格下限上浮一个网格
                buy_con_2 = dat.close[0] < self.buy_price + self.grid_gap
                add_con_3 = dat.close[0] < sma[0]
                if (buy_con_1 and buy_con_2) or add_con_3:
                    # 执行买入，数量：self.p.trade_size，价格：次日开盘价，限价交易。
                    self.buy(data=dat, size=self.trade_size)
                    # 设置self.buy_price价格为次日开盘价下浮一个网格距离
                    self.buy_price = dat.close[0] + self.grid_gap
            # 如果未空仓，可能执行加仓或减仓操作
            else:
                # 加仓条件
                # 条件1：当前收盘价跌破前次买入价下浮一个网格
                add_con_1 = dat.close[0] < self.buy_price
                # 条件2；当前收盘价没有跌破网格下限下浮20%范围
                add_con_2 = dat.close[0] > self.grid_floor * 0.8
                add_con_3 = dat.close[0] < sma[0]

                if add_con_1 and add_con_2:
                    # 执行买入，数量：self.p.trade_size，价格：次日开盘价，限价交易
                    self.buy(data=dat, size=self.trade_size, price=dat.open[0], exectype=bt.Order.Limit)
                    # 设置self.buy_price价格为次日开盘价下移self.params.grid_gap
                    self.buy_price = dat.open[0] - self.grid_gap
                    self.sell_price = self.grid_top
                # 减仓条件：当前收盘价大于设置的网格上限价格
                elif dat.close[0] > self.sell_price or add_con_3:
                    # 执行卖出，数量：self.p.trade_size，价格：次日开盘价，限价交易
                    self.sell(data=dat, size=self.trade_size, price=dat.open[1], exectype=bt.Order.Limit)
                    # 减仓一次，卖出条件上移一个网格
                    self.sell_price = dat.open[0] + self.grid_gap
