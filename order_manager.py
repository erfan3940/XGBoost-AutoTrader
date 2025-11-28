import MetaTrader5 as mt5

def compute_sl_tp(price, direction, config):
    
    # raw percentage calculation
    if direction == "buy":
        sl = price * (1 - config['sl_pct'])
        tp = price * (1 + config['tp_pct'])
    else:  # sell
        sl = price * (1 + config['sl_pct'])
        tp = price * (1 - config['tp_pct'])
    # # broker constraints
    si = mt5.symbol_info('XAUUSD.ecn')
    # point = si.point
    # min_dist = si.trade_stops_level * point
    # buffer = 0.1 * point   # extra safety buffer

    # if direction == "buy":
    #     if sl > price - min_dist:
    #         sl = price - min_dist - buffer
    #     if tp < price + min_dist:
    #         tp = price + min_dist + buffer
    # else:  # sell
    #     if sl < price + min_dist:
    #         sl = price + min_dist + buffer
    #     if tp > price - min_dist:
    #         tp = price - min_dist - buffer

    # round
    sl = round(sl, si.digits)
    tp = round(tp, si.digits)
    return sl, tp

def send_order(entry_price, direction, confidence, config):
    sl, tp = compute_sl_tp(entry_price, direction, config)
    if direction == "buy" :
        buy_request = {
           'action' : mt5.TRADE_ACTION_DEAL,
           'symbol' : config['symbol'],
           'volume' : config['lot'], # float
           'type' : mt5.ORDER_TYPE_BUY,
            #mt5.positions_get().ticket,
           'price' : entry_price,
           'tp' : tp,
           'sl' : sl,
           'deviation' : 20,
           'magic' : config['magic'],
           'comment' : f"ML:{direction}| {confidence*100:.0f}",
           'time_type' : mt5.ORDER_TIME_GTC,
           'type_filling' : mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(buy_request)
        #print(f"@@ sl : {sl} tp : {tp}")
    else:
        sell_request = {
           'action' : mt5.TRADE_ACTION_DEAL,
           'symbol' : config['symbol'],
           'volume' : config['lot'], # float
           'type' : mt5.ORDER_TYPE_SELL,
           'price' : entry_price,
           'tp' : tp,
           'sl' : sl,
           'deviation' : 20,
           'magic' : config['magic'],
           'comment' : f"ML:{direction}| {confidence*100:.0f}",
           'time_type' : mt5.ORDER_TIME_GTC,
           'type_filling' : mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(sell_request)
        
    return result
        
    