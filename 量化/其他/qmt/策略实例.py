#coding:gbk

#导入常用库
import pandas as pd
import numpy as np
import talib
#示例说明：本策略，通过计算快慢双均线，在金叉时买入，死叉时做卖出 点击回测运行 主图选择要交易的股票品种

'''
passorder(
    2 #opType 操作号
    , 1101 #orderType 组合方式
    , '1000044' #accountid 资金账号
    , 'cu2403.SF' #orderCode 品种代码
    , 14 #prType 报价类型
    , 0.0 #price 价格
    , 2 #volume 下单量
    , '示例下单' #strategyName 策略名称
    , 1 #quickTrade 快速下单标记
    , '投资备注' #userOrderId 投资备注
    , C #ContextInfo 策略上下文
)
'''


# 在策略交易界面运行时，account的值会被赋值为策略配置中的账号，
# 编辑器界面运行时，需要手动赋值；编译器环境里执行的下单函数不会产生实际委托
# 回测模式资金账号可以填任意字符串
account = 'test'


def f(ContextInfo):
    print('hello world')
	
def init(C):
	#自定义参数，非调参数时。取最新
	#只能在init/handlebar等有C上下文的函数中获取
	print('自定义参数customize_param：',customize_param)

	#line1和line2分别为两条均线期数
	C.line1=10   #快线参数
	C.line2=20   #慢线参数
	#init handlebar函数的入参是ContextInfo对象 可以缩写为C
	#设置测试标的为主图品种
	C.stock= C.stockcode + '.' +C.market

	#当前是否为回测模式
	C.do_back_test
	
	C.get_last_volume("000001.SZ")# 获取最新流通股本
	C.get_total_share("000001.SZ")#获取总股数
	C.get_trading_dates('600000.SH',start_date='',end_date='',count=30,period='1d')#获取交易日信息


	
	# 初始资金
	C.capital
	C.period
	#获取当前运行到 K 线索引号
	C.barpos
	#获取回测基准标的
	C.benchmark
	
	#非回测时，为-1
	print('回测区间',C.start,C.end)
	
	
	
	
	#是否为最后一根K线
	C.is_last_bar()
	
	# 某根 K 线的第一个 tick 数据到来时，判定该 K 线为新的 K 线，其后的tick不会认为是新的 K 线
	C.is_new_bar()
	
	#根据代码获取名称
	C.get_stock_name('000001.SZ')
	
	# 根据代码返回对应股票的上市时间
	C.get_open_date('000001.SZ')
	
	#'5nSecond'表示每5秒运行1次回调函数,'5nDay'表示每5天运行一次回调函数,'500nMilliSecond'表示每500毫秒运行1次回调函数
	#C.run_time("f","5nSecond","2019-10-14 13:20:00")
	
	download_history_data("000002.SZ","1d","20230101","")
	#在init中运行时（不论subscribe是任何值）仅能取到本地数据
	data=C.get_market_data_ex(
		fields=['close'], 
		stock_code=['000002.SZ'], 
		period='follow', #"tick","1m"：1分钟线,"1h"小时线,"1d"：日线
		start_time='20241219', #包含_time这一时间
		end_time='20241220', #格式为 %Y%m%d 或 %Y%m%d%H%M%S,包含end_time这一时间
		count=-1, #数据个数
		dividend_type='follow', #'none'：不复权,'front':前复权,'back':后复权,'front_ratio': 等比前复权,'back_ratio': 等比后复权
		fill_data=True, #是否填充数据
		subscribe=False#订阅数据开关，默认为True，设置为False时不做数据订阅，只读取本地已有数据。
	)
	
	
	#获取实时净值
	#get_etf_iopv("510050.SH")
	
	#C.get_full_tick(stock_code=['000001.SZ']) # 获取全推数据(只能取最新的分笔，不能取历史分笔)
	
def handlebar(C):
	nowDate=C.get_bar_timetag(C.barpos)
	print('当前交易日',timetag_to_datetime(nowDate, "%Y-%m-%d %H:%M:%S"))

	#当前k线日期
	bar_date = timetag_to_datetime(nowDate, '%Y%m%d%H%M%S')
	#回测不需要订阅最新行情使用本地数据速度更快 指定subscribe参数为否. 如果回测多个品种 需要先下载对应周期历史数据 
	local_data = C.get_market_data_ex(['close'], [C.stock], end_time = bar_date, period = C.period, count = max(C.line1, C.line2), subscribe = False)
	close_list = list(local_data[C.stock].iloc[:, 0])
	#将获取的历史数据转换为DataFrame格式方便计算
	#如果目前未持仓，同时快线穿过慢线，则买入8成仓位
	if len(close_list) <1:
		print(bar_date, '行情不足 跳过')
	line1_mean = round(np.mean(close_list[-C.line1:]), 2)
	line2_mean = round(np.mean(close_list[-C.line2:]), 2)
	print(f"{bar_date} 短均线{line1_mean} 长均线{line2_mean}")
	
	
	
	#m_dAvailable#可用金额
	#m_dBalance#总资产
	#m_dPreBalance#期初权益，指期初时账户的资金金额
	accounts = get_trade_detail_data(account, 'stock', 'account')
	accounts = accounts[0]
	available_cash = int(accounts.m_dAvailable)
	
	
	
	holdings = get_trade_detail_data(account, 'stock', 'position')
	#m_strInstrumentID合约代码
	#m_strExchangeID 交易所
	#m_nVolume当前拥股/持仓量
	#m_dMarketValue市值/合约价值
	#m_nCanUseVolume可用余额
	#m_nFrozenVolume冻结数量
	holdings = {i.m_strInstrumentID + '.' + i.m_strExchangeID : i.m_nVolume for i in holdings}
	holding_vol = holdings[C.stock] if C.stock in holdings else 0
	
	#0：曲线
	#42：柱状线

	# index	number	显示索引位置	填 -1 表示按主图索引显示
	
	# blue：蓝色
	# brown：棕
	# cyan：蓝绿
	# green：绿
	# magenta：品红
	# red：红
	# white：白
	# yellow：黄
	C.paint("line1_mean",line1_mean,-1,42, 'white','noaxis')
	C.paint("line2_mean",line2_mean,-1,0, 'blue','noaxis')

	if holding_vol == 0 and line1_mean > line2_mean:
		#1：椭圆
		#0：矩形
		C.draw_icon(True, close_list[-1], 1)

		vol = int(available_cash / close_list[-1] / 100) * 100
		#下单开仓
		passorder(23, 1101, account, C.stock, 5, -1, vol, C)
		print(f"{bar_date} 开仓")
		C.draw_text(True, close_list[-1]+1, '开')
	#如果目前持仓中，同时快线下穿慢线，则全部平仓
	elif holding_vol > 0 and line1_mean < line2_mean:
		C.draw_icon(True, close_list[-1], 0)
		#下单平仓
		passorder(24, 1101, account, C.stock, 5, -1, holding_vol, C)
		print(f"{bar_date} 平仓")
		C.draw_text(True, close_list[-1]+1, '平')


def stop(ContextInfo):
    print( 'strategy is stop !')