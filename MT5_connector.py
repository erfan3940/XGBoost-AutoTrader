import MetaTrader5 as mt5
import config

def initialize_MT5():
    if not mt5.initialize(login = config.LOGIN,password = config.PASSWORD,server = config.SERVER):
        print(f"initialize() failed, error code = {mt5.last_error()}")
        return False
    print("MT5 initialized successfully.")
    account_info = mt5.account_info()
    if account_info is None:
        print("Failed to get account info.")
        return False

    print(f"Login: {account_info.login}")
    print(f"Balance: {account_info.balance}, Equity: {account_info.equity}")
    return True

def shutdown_mt5():
    mt5.shutdown()
    print("MT5 connection shut down.") 
