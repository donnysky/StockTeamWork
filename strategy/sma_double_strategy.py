
# 导入backtrader平台
import backtrader as bt
import streamlit as st

# --------------------------
# 全局配置和工具函数 双sma交叉策略
# --------------------------
class SmaDoubleStrategy(bt.Strategy):
    """基础回测策略模板"""
    params = (
        ('maperiod', 15),
    )

    def __init__(self, trade_base):
        # self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        # 定义两条移动平均线
        self.smas = {d._name: bt.indicators.SMA(d.close, period=trade_base["ma_period"]) for d in self.datas}
        self.sma_shorts = {d._name: bt.indicators.SMA(d.close, period=trade_base["ma_5"]) for d in self.datas}
        self.sma_longs = {d._name: bt.indicators.SMA(d.close, period=trade_base["ma_10"]) for d in self.datas}
        # 金叉/死叉信号
        # self.crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)
        self.crossovers = {d._name: bt.indicators.CrossOver((bt.indicators.SMA(d.close, period=trade_base["ma_5"])), (bt.indicators.SMA(d.close, period=trade_base["ma_10"]))) for d in self.datas}

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行, 价格: {order.executed.price:.2f}, 数量: {order.executed.size:.2f}, 成本: {order.executed.value:.2f}, 佣金: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            if order.issell():
                self.log(f'卖出执行, 价格: {order.executed.price:.2f}, 数量: {order.executed.size:.2f}, 成本: {order.executed.value:.2f}, 销售额: {order.executed.price*order.executed.size*-1:.2f},  佣金: {order.executed.comm:.2f}')
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/被拒绝')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            # self.log(f'交易收益, 总收益: {trade.value:.2f}, 净收益: {trade.pnlcomm:.2f}')
            return

    # def notify_cashvalue(self, cash, value):
    #     self.log(f'cash:{cash:.2f},value:{value:.2f}')

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        st.session_state.backtest_logs.append(f'{dt.isoformat()} - {txt}')

    def next(self):
        if self.order:
            return
        # 如果没有持仓且出现金叉
        for dat in self.datas:
            position = self.getposition(dat)
            sma_short = self.sma_shorts[dat._name]
            sma_long = self.sma_longs[dat._name]
            crossover = self.crossovers[dat._name]
            if not position:
                if crossover > 0:
                    self.order = self.buy(data=dat)
                    self.log(f'买入信号, 价格: {dat.close[0]:.2f}')
            # 如果持有仓位且出现死叉 or self.dataclose < self.sma
            else:
                if crossover < 0 or (dat.close[0] < dat.close[-1] < dat.close[-2]):
                    self.order = self.sell(data=dat)
                    self.log(f'卖出信号, 价格: {dat.close[0]:.2f}')

    def stop(self):
        self.log('(双线策略：) 期末总资金 %.2f' % (self.broker.getvalue()))
        # self.log(f'stop...')
        # if self.position:
        #     self.sell(percents=100)
        #     self.log(f'stop, position: {self.position}')

