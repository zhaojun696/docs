#coding:gbk


# 运行周期分笔

account='test'
import datetime
def init(C):
    C.acct = account
    C.accoutType = 'STOCK'
    
    print('当前账号',C.acct)
    C.buy_code = 23 if C.accoutType.lower() == 'stock' else 33

finish = False
current_date = ''
def handlebar(C):
    global finish, current_date
    now = datetime.datetime.now()
    now_time = now.strftime('%H%M')
    if now_time < '0931' or now_time > '1800':
        return
    bar_date = now.strftime('%Y%m%d')
    if current_date != bar_date:
        finish = False
    
    current_date = bar_date
    if finish:
        return
    ipoStock=get_ipo_data("STOCK")#返回新股信息
    print('当日新股', ipoStock)
    limit_info = get_new_purchase_limit(C.acct)
    print('账户限额', limit_info)
    #print(ipoStock)
    stock_volume_dict = {i : ipoStock[i]['maxPurchaseNum'] for i in ipoStock}
    stock_price_dict = {i : ipoStock[i]['issuePrice'] for i in ipoStock}
    for stock in stock_volume_dict:
        market = stock[-2:]
        if market not in limit_info:
            print(market, limit_info, '缺少限制信息')
            continue
        stock_volume_dict[stock] = min(stock_volume_dict[stock], limit_info[market])
    print('新股可申字典', stock_volume_dict)
    for stock in stock_volume_dict:
        if stock_volume_dict[stock] <=0:
            print(f"{stock} {C.get_stock_name(stock)} 可申购数量不大于0 跳过申购")
            continue
        passorder(C.buy_code,1101, C.acct, stock,11,stock_price_dict[stock], stock_volume_dict[stock],'新股申购',2,stock,C)
        print(f"新股申购 {stock} {stock_volume_dict[stock]}股")
    
    ipobond=get_ipo_data("BOND")#返回新债信息
    print('当日新债', ipobond)
    bond_volume_dict = {i : ipobond[i]['maxPurchaseNum'] for i in ipobond}
    bond_price_dict = {i : ipobond[i]['issuePrice'] for i in ipobond}
    print('新债可申字典', bond_volume_dict)
    for bond in bond_volume_dict:
        if(bond_volume_dict[bond]<=0):
            print(f"{bond}  可申购数量不大于0 跳过申购")
            continue

        passorder(C.buy_code,1101, C.acct, bond,11,bond_price_dict[bond], bond_volume_dict[bond],'新债申购',2,bond,C)
        print(f"新债申购 {bond} {bond_volume_dict[bond]}张")
    finish = True