import streamlit as st
# 导入backtrader平台
import backtrader as bt


# --------------------------
# 全局配置和工具函数
# --------------------------
class SmaStrategy(bt.Strategy):
    """基础回测策略模板"""
    params = (
        ('maperiod', 15),
    )

    def __init__(self, trade_base):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.trade_base = trade_base
        # self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=trade_base["ma_period"])
        # 为每个数据源创建 SMA 指标
        self.smas = {d._name: bt.indicators.SMA(d.close, period=trade_base["ma_period"]) for d in self.datas}
        print(trade_base["ma_period"])

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'买入执行, 价格: {order.executed.price:.2f}, 成本: {order.executed.value:.2f}, 佣金: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(
                    f'卖出执行, 价格: {order.executed.price:.2f}, 收入: {order.executed.value:.2f}, 佣金: {order.executed.comm:.2f}')
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/被拒绝')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        # self.log(f'交易收益, 总收益: {trade.pnl:.2f}, 净收益: {trade.pnlcomm:.2f}')

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        st.session_state.backtest_logs.append(f'{dt.isoformat()} - {txt}')

    def next(self):
        if self.order:
            return

        for dat in self.datas:
            position = self.getposition(dat)
            sma = self.smas[dat._name]
            if not position:
                # print(str(self.dataclose[0]) + "____" + str(self.dataclose[-1]))
                if dat.close[0] > sma[0]:
                    self.log(f'买入信号, 价格: {dat.close[0]:.2f}')
                    self.order = self.buy(data=dat)
            else:
                if dat.close[0] < sma[0]:
                    self.log(f'卖出信号, 价格: {dat.close[0]:.2f}')
                    self.order = self.sell(data=dat)

        # if not self.position:
        #     print(str(self.dataclose[0]) + "____" + str(self.dataclose[-1]))
        #     if self.dataclose[0] > self.sma[0] and self.dataclose[0] > self.dataclose[-1]:
        #         self.log(f'买入信号, 价格: {self.dataclose[0]:.2f}')
        #         self.order = self.buy()
        # else:
        #     if self.dataclose[0] < self.sma[0]:
        #         self.log(f'卖出信号, 价格: {self.dataclose[0]:.2f}')
        #         self.order = self.sell()
