
# 导入backtrader平台
import backtrader as bt

# 打板策略,当日收盘价涨停时买入（做多），当收盘价下跌超5%卖出（做空）
class UpperStrategy(bt.Strategy):
    params=(('maperiod',5),
            ('printlog',True),)
    def __init__(self):
        #指定价格序列
        self.log(f'收盘价, {len(self.datas)}')
        self.dataclose=self.datas[0].close
        # 初始化交易指令、买卖价格和手续费
        self.order = None
        self.buyprice = None
        self.buycomm = None
#         if self.p.sizer is not None:
#             self.sizer = self.p.sizer
        #添加移动均线指标
        # self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.maperiod)
    #策略核心，根据条件执行买卖交易指令（必选）连续三日上涨或是前一日的10%及以上
    def next(self):
        # 记录收盘价
        self.log(f'收盘价, {self.dataclose[0]}')
        if self.order: # 检查是否有指令等待执行,
            return
        # 检查是否持仓
        print("现金："+str(self.broker.getcash())+" 买需要的钱:"+str(10000*self.dataclose[0]))
        # 没有持仓 self.p.sizer.stake
        if ((not self.position) or (self.broker.getcash() >= 10000*self.dataclose[0])):
            #执行买入条件判断：收盘价格上涨突破15日均线
            print("self.dataclose[0]"+str(self.dataclose[0])+"self.dataclose[-1]"+str(self.dataclose[-1])+" "+str(self.dataclose[0] > self.dataclose[-1]))
            print("self.dataclose[-1]"+str(self.dataclose[-1])+"self.dataclose[-2]"+str(self.dataclose[-2])+" "+str(self.dataclose[-1] > self.dataclose[-2]))
            if self.dataclose[0] > self.dataclose[-1]:
                if self.dataclose[-1] > self.dataclose[-2]:
                    self.log('BUY CREATE 3 UP, %.2f' % self.dataclose[0])
                    #执行买入
                    self.order = self.buy()
            elif self.dataclose[0] >= (self.dataclose[-1]*110/100):
                self.log('BUY CREATE UPPER, %.2f' % self.dataclose[0])
                #执行买入
                self.order = self.buy()
            else:
                print("buy nothing but sell.......")
                print("卖出条件判断："+str(self.dataclose[0]/self.dataclose[-1])+"%")
                if self.position and ((self.dataclose[0] <= (self.dataclose[-1]*95/100)) or (self.dataclose[0] <= (self.dataclose[-2]*95/100)) or (self.dataclose[0] <= (self.dataclose[-3]*95/100))):
                    self.log('SELL CREATE, %.2f' % self.dataclose[0])
                    #执行卖出
                    self.order = self.sell()
        else:
            #执行卖出条件判断：收盘价格跌破15日均线
            print("卖出条件判断："+str(self.dataclose[0]/self.dataclose[-1])+"%"+" or "+str(self.dataclose[0]/(self.dataclose[-2]))+"%")
            if ((self.dataclose[0] <= (self.dataclose[-1]*95/100)) or (self.dataclose[0] <= (self.dataclose[-2]*95/100)) or (self.dataclose[0] <= (self.dataclose[-3]*95/100))):
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                #执行卖出
                self.order = self.sell()
    #交易记录日志（可省略，默认不输出结果）
    def log(self, txt, dt=None,doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()},{txt}')
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
        self.log(f'策略收益：毛收益 {trade.pnl:.2f}, 净收益 {trade.pnlcomm:.2f}')
    #回测结束后输出结果（可省略，默认输出结果）
    def stop(self):
        self.log('(MA均线： %2d日) 期末总资金 %.2f' %(self.params.maperiod, self.broker.getvalue()), doprint=True)