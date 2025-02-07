#%%
# https://dict.thinktrader.net/nativeApi/download_xtquant.html
#%%
from xtquant import xtdata
stocks = ['000001.SZ', '600000.SH']
xtdata.download_history_data(stocks[0], '1d')

xtdata.get_market_data_ex(stock_list=['000001.SZ'])['000001.SZ']

#%%
xtdata.get_instrument_detail('000001.SZ')

#%%
import time
period = "1m"

codes=["000001.SZ"]
def do_subscribe_quote(stock_list:list, period:str):
  for s in stock_list:
    i=xtdata.subscribe_quote(s,period = period)
    print(i)
  time.sleep(1) # 等待订阅完成
do_subscribe_quote(codes,period=period)

def do_download_history(stock_list:list, period:str):
  for s in stock_list:
    xtdata.download_history_data(s, period)
    print('下载历史数据')
do_download_history(codes,period=period)

xtdata.get_market_data_ex([],codes,start_time='20240101',period=period,count=2)

#%%

xtdata.get_full_tick(['000001.SZ'])
# 全推数据只有分笔周期，每次增量推送数据有变化的品种
xtdata.subscribe_whole_quote(['SH'])
#%%
xtdata.download_financial_data(["000001.SZ","600519.SH","430017.BJ"], table_list=["Balance","Income","CashFlow",
                                    "Capital","PershareIndex","Top10flowholder","Holdernum"])
xtdata.get_financial_data(["000001.SZ","600519.SH","430017.BJ"],["Balance","Income","CashFlow",
                                    "Capital","PershareIndex","Top10flowholder","Holdernum"])['000001.SZ']['Top10flowholder']


#%%
# 基金列表
ret_sector_data = xtdata.get_stock_list_in_sector('沪深基金') [:10]
ret_sector_data


# 使用回调时，必须要同时使用xtdata.run()来阻塞程序，否则程序运行到最后一行就直接结束退出了。
xtdata.run()
