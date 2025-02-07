from xtquant import xtconstant
def nihuigou(xt_trader,acc,symbol="131810.SZ"):
    asset = xt_trader.query_stock_asset(acc)
    if asset.cash >1000:
        vol = int((asset.cash//1000)*10)
        async_seq = xt_trader.order_stock_async(acc,symbol,xtconstant.STOCK_SELL,vol,xtconstant.LATEST_PRICE,9.99,strategy_name='nihuigou',order_remark='nihuigou卖出')
        print(f'nihuigou下单：async_seq({async_seq})卖出逆回购：{vol}手')
    else:
        print("nihuigou下单：逆回购金额不足")
