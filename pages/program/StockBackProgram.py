import baostock as bs
import pandas as pd


def download_trade_date(trade_file_name: str, start_date_in: str, end_date_in: str):
    '''
    获取交易日信息
    :param trade_file_name: csv file
    :param start_date_in: yyyy-MM-dd
    :param end_date_in: yyyy-MM-dd
    :return:
    '''
    #### 登陆系统 ####
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)

    #### 获取交易日信息 ####
    rs = bs.query_trade_dates(start_date=start_date_in, end_date=end_date_in)
    print('query_trade_dates respond error_code:' + rs.error_code)
    print('query_trade_dates respond  error_msg:' + rs.error_msg)

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    result['month'] = result['calendar_date'].str[:7]
    #### 结果集输出到csv文件 ####
    result.to_csv(trade_file_name, encoding="utf-8", index=False)
    print(result)

    #### 登出系统 ####
    bs.logout()


def read_csv(file_name: str):
    df = pd.read_csv("./data/" + file_name)
    return df


def download_all_stocks_by_day(trade_date_file: str, trade_date: str, all_stock_file: str):
    """
    获取某日所有证券信息,第一个参数交易文件交易日期默认从这文件里取第一个交易日期，第二个参数交易日期如果给了优先用这个
    """
    print(trade_date_file)
    #### 登陆系统 ####
    lg = bs.login();
    # 显示登陆返回信息
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)

    #### 获取某日所有证券信息 ####
    # 当参数“day”为空时，默认取当天日期。闭市后日K线数据更新，该接口才会返回当天数据，否则返回空。
    day_in = trade_date
    if trade_date is None:
        day_in = "2024-01-02"
    rs = bs.query_all_stock(day=day_in)
    print('query_all_stock respond error_code:' + rs.error_code)
    print('query_all_stock respond  error_msg:' + rs.error_msg)

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    #### 结果集输出到csv文件 ####
    result.to_csv("./data/" + all_stock_file, encoding="utf-8", index=False)
    print(result)
    #### 登出系统 ####
    bs.logout()


def download_stocks_shz50(stock_file: str, query_date: str):
    """
    stock_file: 保存文件名
    date：查询日期，格式XXXX-XX-XX，为空时默认最新日期。
    updateDate	code	code_name
    2018-11-26	sh.600000	浦发银行
    """
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)

    # 获取上证50成分股
    rs = bs.query_sz50_stocks(query_date)
    print('query_sz50 error_code:' + rs.error_code)
    print('query_sz50  error_msg:' + rs.error_msg)

    # 打印结果集
    sz50_stocks = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        sz50_stocks.append(rs.get_row_data())
    result = pd.DataFrame(sz50_stocks, columns=rs.fields)
    # 结果集输出到csv文件
    result.to_csv("./data/" + stock_file, encoding="utf-8", index=False)
    print(result)

    # 登出系统
    bs.logout()


def download_stocks_hs300(stock_file: str, query_date: str):
    """
    stock_file: 保存文件名
    date：查询日期，格式XXXX-XX-XX，为空时默认最新日期。
    updateDate	code	code_name
    2018-11-26	sh.600000	浦发银行
    """
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)

    # 获取沪深300成分股
    rs = bs.query_hs300_stocks(query_date)
    print('query_hs300 error_code:' + rs.error_code)
    print('query_hs300  error_msg:' + rs.error_msg)

    # 打印结果集
    hs300_stocks = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        hs300_stocks.append(rs.get_row_data())
    result = pd.DataFrame(hs300_stocks, columns=rs.fields)
    # 结果集输出到csv文件
    result.to_csv("./data/" + stock_file, encoding="utf-8", index=False)
    print(result)

    # 登出系统
    bs.logout()


def download_stock_industry(stock_file: str, query_date: str):
    """
    获取行业分类数据
    stock_file: 保存文件名
    date：查询日期，格式XXXX-XX-XX，为空时默认最新日期。
    updateDate	code	code_name	industry	industryClassification
    2018-11-26	sh.600000	浦发银行	J66货币金融服务	证监会行业分类
    """
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)

    # 获取行业分类数据
    rs = bs.query_stock_industry("", query_date)
    # rs = bs.query_stock_basic(code_name="浦发银行")
    print('query_stock_industry error_code:' + rs.error_code)
    print('query_stock_industry respond  error_msg:' + rs.error_msg)

    # 打印结果集
    industry_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        industry_list.append(rs.get_row_data())
    result = pd.DataFrame(industry_list, columns=rs.fields)
    # 结果集输出到csv文件
    result.to_csv("./data/" + stock_file, encoding="utf-8", index=False)
    print(result)

    # 登出系统
    bs.logout()


def download_stock_basic(stock_file: str, stock_code: str):
    """
    方法说明：通过API接口获取证券基本资料，可以通过参数设置获取对应证券代码、证券名称的数据。 返回类型：pandas的DataFrame类型。
    stock_file: 保存文件名
    参数含义：
    code：A股股票代码，sh或sz.+6位数字代码，或者指数代码，如：sh.601398。sh：上海；sz：深圳。可以为空；
    code_name：股票名称，支持模糊查询，可以为空。
    当参数为空时，输出全部股票的基本信息。
    返回：
    code	code_name	ipoDate	outDate	type	status
    sh.600000	浦发银行	1999-11-10		1	1

    返回数据说明
    参数名称	参数描述
    code	证券代码
    code_name	证券名称
    ipoDate	上市日期
    outDate	退市日期
    type	证券类型，其中1：股票，2：指数，3：其它，4：可转债，5：ETF
    status	上市状态，其中1：上市，0：退市
    """
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)

    # 获取证券基本资料
    rs = bs.query_stock_basic(code=stock_code)
    # rs = bs.query_stock_basic(code_name="浦发银行")  # 支持模糊查询
    print('query_stock_basic respond error_code:' + rs.error_code)
    print('query_stock_basic respond  error_msg:' + rs.error_msg)

    # 打印结果集
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    # 结果集输出到csv文件
    result.to_csv("./data/" + stock_file, encoding="utf-8", index=False)
    print(result)

    # 登出系统
    bs.logout()


def download_index_k_data(stock_file: str, stock_code: str, start_date: str, end_date: str, frequency: str):
    """
    通过API接口获取指数(综合指数、规模指数、一级行业指数、二级行业指数、策略指数、成长指数、价值指数、主题指数)K线数据,d。
    指数未提供分钟线数据。
     1. 综合指数，例如：sh.000001 上证指数，sz.399106 深证综指 等；
     2. 规模指数，例如：sh.000016 上证50，sh.000300 沪深300，sh.000905 中证500，sz.399001 深证成指等；
     3. 一级行业指数，例如：sh.000037 上证医药，sz.399433 国证交运 等；
     4. 二级行业指数，例如：sh.000952 300地产，sz.399951 300银行 等；
     5. 策略指数，例如：sh.000050 50等权，sh.000982 500等权 等；
     6. 成长指数，例如：sz.399376 小盘成长 等；
     7. 价值指数，例如：sh.000029 180价值 等；
     8. 主题指数，例如：sh.000015 红利指数，sh.000063 上证周期 等；
     9. 基金指数，例如：sh.000011 上证基金指数 等；
     10. 债券指数，例如：sh.000012 上证国债指数 等；

     参数含义：
    code：股票代码，sh或sz.+6位数字代码，或者指数代码，如：sh.601398。sh：上海；sz：深圳。此参数不可为空；
    fields：指示简称，支持多指标输入，以半角逗号分隔，填写内容作为返回类型的列。详细指标列表见历史行情指标参数章节 。此参数不可为空；
    start：开始日期（包含），格式“YYYY-MM-DD”，为空时取2015-01-01；
    end：结束日期（不包含），格式“YYYY-MM-DD”，为空时取最近一个交易日；
    frequency：数据类型，默认为d，日k线；d=日k线、w=周、m=月、5=5分钟、15=15分钟、30=30分钟、60=60分钟k线数据，不区分大小写；指数没有分钟线数据；周线每周最后一个交易日才可以获取，月线第月最后一个交易日才可以获取。

    返回数据说明
    参数名称	参数描述	说明
    date	交易所行情日期	格式：YYYY-MM-DD
    code	证券代码	格式：sh.600000。sh：上海，sz：深圳
    open	今开盘价格	精度：小数点后4位；单位：人民币元
    high	最高价	精度：小数点后4位；单位：人民币元
    low	最低价	精度：小数点后4位；单位：人民币元
    close	今收盘价	精度：小数点后4位；单位：人民币元
    preclose	昨日收盘价	精度：小数点后4位；单位：人民币元
    volume	成交数量	单位：股
    amount	成交金额	精度：小数点后4位；单位：人民币元
    pctChg	涨跌幅	精度：小数点后6位
    """
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)

    # 详细指标参数，参见“历史行情指标参数”章节；“周月线”参数与“日线”参数不同。
    # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
    rs = bs.query_history_k_data_plus(stock_code,
                                      "date,code,open,high,low,close,preclose,volume,amount,pctChg",
                                      start_date=start_date, end_date=end_date, frequency=frequency)
    print('query_history_k_data_plus respond error_code:' + rs.error_code)
    print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)

    # 打印结果集
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    # 结果集输出到csv文件
    result.to_csv("./data/" + stock_file, encoding="utf-8", index=False)
    print(result)

    # 登出系统
    bs.logout()


def download_stock_kday_data(stock_file: str, stock_code: str, start_date: str, end_date: str, frequency: str):
    """
    方法说明：通过API接口获取A股历史交易数据，可以通过参数设置获取日k线、周k线、月k线，以及5分钟、15分钟、30分钟和60分钟k线数据，适合搭配均线数据进行选股和分析。

     参数含义：
    code：股票代码，sh或sz.+6位数字代码，或者指数代码，如：sh.601398。sh：上海；sz：深圳。此参数不可为空；
    fields：指示简称，支持多指标输入，以半角逗号分隔，填写内容作为返回类型的列。详细指标列表见历史行情指标参数章节，日线与分钟线参数不同。此参数不可为空；
    start：开始日期（包含），格式“YYYY-MM-DD”，为空时取2015-01-01；
    end：结束日期（包含），格式“YYYY-MM-DD”，为空时取最近一个交易日；
    frequency：数据类型，默认为d，日k线；d=日k线、w=周、m=月、5=5分钟、15=15分钟、30=30分钟、60=60分钟k线数据，不区分大小写；指数没有分钟线数据；周线每周最后一个交易日才可以获取，月线每月最后一个交易日才可以获取。
    adjustflag：复权类型，默认不复权：3；1：后复权；2：前复权。已支持分钟线、日线、周线、月线前后复权。 BaoStock提供的是涨跌幅复权算法复权因子，具体介绍见BaoStock复权因子简介。

    返回数据说明
    参数名称	参数描述	算法说明
    date	交易所行情日期
    code	证券代码
    open	开盘价
    high	最高价
    low	最低价
    close	收盘价
    preclose	前收盘价	见表格下方详细说明
    volume	成交量（累计 单位：股）
    amount	成交额（单位：人民币元）
    adjustflag	复权状态(1：后复权， 2：前复权，3：不复权）
    turn	换手率	[指定交易日的成交量(股)/指定交易日的股票的流通股总股数(股)]*100%
    tradestatus	交易状态(1：正常交易 0：停牌）
    pctChg	涨跌幅（百分比）	日涨跌幅=[(指定交易日的收盘价-指定交易日前收盘价)/指定交易日前收盘价]*100%
    peTTM	滚动市盈率	(指定交易日的股票收盘价/指定交易日的每股盈余TTM)=(指定交易日的股票收盘价*截至当日公司总股本)/归属母公司股东净利润TTM
    pbMRQ	市净率	(指定交易日的股票收盘价/指定交易日的每股净资产)=总市值/(最近披露的归属母公司股东的权益-其他权益工具)
    psTTM	滚动市销率	(指定交易日的股票收盘价/指定交易日的每股销售额)=(指定交易日的股票收盘价*截至当日公司总股本)/营业总收入TTM
    pcfNcfTTM	滚动市现率	(指定交易日的股票收盘价/指定交易日的每股现金流TTM)=(指定交易日的股票收盘价*截至当日公司总股本)/现金以及现金等价物净增加额TTM
    isST	是否ST股，1是，0否
    """
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)

    #### 获取沪深A股历史K线数据 ####
    # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
    # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
    # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
    rs = bs.query_history_k_data_plus(stock_code,
                                      "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
                                      start_date=start_date, end_date=end_date,
                                      frequency=frequency, adjustflag="3")
    print('query_history_k_data_plus respond error_code:' + rs.error_code)
    print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    #### 结果集输出到csv文件 ####
    result.to_csv("./data/day_" + stock_file, encoding="utf-8", index=False)
    print(result)

    # 登出系统
    bs.logout()


def download_stock_kmin_data(stock_file: str, stock_code: str, start_date: str, end_date: str, frequency: str):
    """
    方法说明：通过API接口获取A股历史交易数据，可以通过参数设置获取日k线、周k线、月k线，以及5分钟、15分钟、30分钟和60分钟k线数据，适合搭配均线数据进行选股和分析。

     参数含义：
    code：股票代码，sh或sz.+6位数字代码，或者指数代码，如：sh.601398。sh：上海；sz：深圳。此参数不可为空；
    fields：指示简称，支持多指标输入，以半角逗号分隔，填写内容作为返回类型的列。详细指标列表见历史行情指标参数章节，日线与分钟线参数不同。此参数不可为空；
    start：开始日期（包含），格式“YYYY-MM-DD”，为空时取2015-01-01；
    end：结束日期（包含），格式“YYYY-MM-DD”，为空时取最近一个交易日；
    frequency：数据类型，默认为d，日k线；d=日k线、w=周、m=月、5=5分钟、15=15分钟、30=30分钟、60=60分钟k线数据，不区分大小写；指数没有分钟线数据；周线每周最后一个交易日才可以获取，月线每月最后一个交易日才可以获取。
    adjustflag：复权类型，默认不复权：3；1：后复权；2：前复权。已支持分钟线、日线、周线、月线前后复权。 BaoStock提供的是涨跌幅复权算法复权因子，具体介绍见BaoStock复权因子简介。

    返回数据说明
    参数名称	参数描述	算法说明
    date	交易所行情日期
    code	证券代码
    open	开盘价
    high	最高价
    low	最低价
    close	收盘价
    preclose	前收盘价	见表格下方详细说明
    volume	成交量（累计 单位：股）
    amount	成交额（单位：人民币元）
    adjustflag	复权状态(1：后复权， 2：前复权，3：不复权）
    turn	换手率	[指定交易日的成交量(股)/指定交易日的股票的流通股总股数(股)]*100%
    tradestatus	交易状态(1：正常交易 0：停牌）
    pctChg	涨跌幅（百分比）	日涨跌幅=[(指定交易日的收盘价-指定交易日前收盘价)/指定交易日前收盘价]*100%
    peTTM	滚动市盈率	(指定交易日的股票收盘价/指定交易日的每股盈余TTM)=(指定交易日的股票收盘价*截至当日公司总股本)/归属母公司股东净利润TTM
    pbMRQ	市净率	(指定交易日的股票收盘价/指定交易日的每股净资产)=总市值/(最近披露的归属母公司股东的权益-其他权益工具)
    psTTM	滚动市销率	(指定交易日的股票收盘价/指定交易日的每股销售额)=(指定交易日的股票收盘价*截至当日公司总股本)/营业总收入TTM
    pcfNcfTTM	滚动市现率	(指定交易日的股票收盘价/指定交易日的每股现金流TTM)=(指定交易日的股票收盘价*截至当日公司总股本)/现金以及现金等价物净增加额TTM
    isST	是否ST股，1是，0否
    """
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)

    #### 获取沪深A股历史K线数据 #### frequency=5
    # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
    # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
    # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
    rs = bs.query_history_k_data_plus(stock_code,
                                      "date,time,code,open,high,low,close,volume,amount,adjustflag",
                                      start_date=start_date, end_date=end_date,
                                      frequency=frequency, adjustflag="3")
    print('query_history_k_data_plus respond error_code:' + rs.error_code)
    print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    #### 结果集输出到csv文件 ####
    result.to_csv("./data/minutes_" + stock_file, encoding="utf-8", index=False)
    print(result)

    # 登出系统
    bs.logout()
