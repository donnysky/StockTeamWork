import streamlit as st
# 导入backtrader平台
import backtrader as bt

# 打板策略,当日收盘价涨停时买入（做多），当收盘价下跌超5%卖出（做空）
class TopBuyStrategy(bt.Strategy):
    params=(('maperiod', 5),
            ('printlog', True),)
    def __init__(self, trade_base):
        #指定价格序列
        self.log("=============================TopBuyStrategy __init__ ==========================")
        self.log(trade_base)
        self.log("=============================TopBuyStrategy end ==========================")
        # self.log(f'收盘价, {len(self.datas)}')
        # self.dataclose=self.datas[0].close
        # 初始化交易指令、买卖价格和手续费
        self.order = None
        self.buyprice = None
        self.buycomm = None
        #添加移动均线指标
        self.smas = {d._name: bt.indicators.SMA(d.close, period=trade_base["ma_period"]) for d in self.datas}

    #策略核心，根据条件执行买卖交易指令（必选）连续三日上涨或是前一日的10%及以上
    def next(self):
        # 记录收盘价
        # self.log(f'收盘价, {self.dataclose[0]}')
        # 检查是否有指令等待执行,
        if self.order:
            return
        # 检查是否持仓
        self.log("现金："+str(self.broker.getcash()))
        for dat in self.datas:
            position = self.getposition(dat)
            # sma = self.smas[dat._name]
            if not position:
                if dat.close[0] > dat.close[-1] > dat.close[-2]:
                    self.order = self.buy(data=dat)
                elif dat.close[0] >= (dat.close[-1] * 110 / 100):
                    self.order = self.buy(data=dat)
            else:
                if ((dat.close[0] <= (dat.close[-1] * 95 / 100)) or (
                        dat.close[0] <= (dat.close[-2] * 95 / 100)) or (
                        dat.close[0] <= (dat.close[-3] * 95 / 100))) or (dat.close[0] < dat.close[-1] < dat.close[-2]):
                    # 执行卖出
                    self.order = self.sell(data=dat)

    #交易记录日志（可省略，默认不输出结果）
    def log(self, txt, dt=None,doprint=False):
        dt = dt or self.datas[0].datetime.date(0)
        st.session_state.backtest_logs.append(f'{dt.isoformat()} - {txt}')

    #记录交易执行情况（可省略，默认不输出结果）
    def notify_order(self, order):
        # 如果order为submitted/accepted,返回空
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 如果order为buy/sell executed,报告价格结果
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入: 价格:{order.executed.price},\
                成本:{order.executed.value},\
                手续费:{order.executed.comm}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(f'卖出: 价格：{order.executed.price},\
                成本: {order.executed.value},\
                手续费{order.executed.comm}')
            self.bar_executed = len(self)
        # 如果指令取消/交易失败, 报告结果
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('交易失败')
        self.order = None

    #记录交易收益情况（可省略，默认不输出结果）
    def notify_trade(self,trade):
        if not trade.isclosed:
            return
        # self.log(f'策略收益：毛收益 {trade.pnl:.2f}, 净收益 {trade.pnlcomm:.2f}')

    #回测结束后输出结果（可省略，默认输出结果）
    def stop(self):
        self.log('(打板策略：) 期末总资金 %.2f' %(self.broker.getvalue()), doprint=True)