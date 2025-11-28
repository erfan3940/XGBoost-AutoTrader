import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from typing import Union, Optional, Literal

def get_mt5_rates(
    symbol: str,
    timeframe,
    request_type: Literal["pos", "from", "range"] = "pos_count",
    pos: Optional[int] = None,
    count: Optional[int] = None,
    from_time: Optional[Union[datetime, str]] = None,
    to_time: Optional[Union[datetime, str]] = None
) -> pd.DataFrame:
    """
    Unified function to retrieve historical data from MetaTrader 5.
    
    Parameters:
    -----------
    symbol : str
        Symbol name (e.g., "EURUSD")
    timeframe : int
        MT5 timeframe constant (e.g., mt5.TIMEFRAME_H1)
    request_type : str
        Type of request: "pos_count", "from_to", or "time_range"
    pos : int, optional
        Starting position/index (0 = current bar) (1 = last closed bar)
    count : int, optional
        Number of bars to retrieve
    from_time : datetime/str, optional
        Start time for time range requests
    to_time : datetime/str, optional
        End time for time range requests
    
    
    Examples:
    ---------
    # Type 1: Copy rates from position (like copy_rates_from_pos)
    df1 = get_mt5_rates("EURUSD", mt5.TIMEFRAME_H1, "pos_count", pos=0, count=100)
    
    # Type 2: Copy rates from-to positions (like copy_rates_from)
    df2 = get_mt5_rates("EURUSD", mt5.TIMEFRAME_D1, "from_to", pos=10, count=50)
    
    # Type 3: Copy rates by time range (like copy_rates_range)
    df3 = get_mt5_rates("EURUSD", mt5.TIMEFRAME_H4, "time_range", 
                       from_time=datetime(2024, 1, 1), 
                       to_time=datetime(2024, 1, 31))
    """
    
    if not mt5.initialize():
        raise ConnectionError("Failed to initialize MT5 connection")
    
    try:
        if request_type == "pos_count":
            # Type 1: Copy rates from position (like copy_rates_from)
            if pos is None or count is None:
                raise ValueError("For 'pos_count' type, both 'pos' and 'count' must be provided")
            rates = mt5.copy_rates_from_pos(symbol, timeframe, pos, count)
            
        elif request_type == "from_to":
            # Type 2: Copy rates from-to positions
            if pos is None or count is None:
                raise ValueError("For 'from_to' type, both 'pos' and 'count' must be provided")
            rates = mt5.copy_rates_from(symbol, timeframe, pos, count)
            
        elif request_type == "time_range":
            # Type 3: Copy rates by time range
            if from_time is None or to_time is None:
                raise ValueError("For 'time_range' type, both 'from_time' and 'to_time' must be provided")
            
            # Convert string inputs to datetime if needed
            if isinstance(from_time, str):
                from_time = datetime.fromisoformat(from_time)
            if isinstance(to_time, str):
                to_time = datetime.fromisoformat(to_time)
                
            rates = mt5.copy_rates_range(symbol, timeframe, from_time, to_time)
            
        else:
            raise ValueError("Invalid request_type. Use 'pos_count', 'from_to', or 'time_range'")
        
        if rates is None or len(rates) == 0:
            print(f"No data retrieved for {symbol} on timeframe {timeframe}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Rename columns for consistency
        df.rename(columns={
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume'
        }, inplace=True)
        
        # Keep only relevant columns that exist
        available_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] 
                         if col in df.columns]
        df = df[available_cols]
        
        return df

    except Exception as E:
        print(f"Error retrieving data: {e}")
        return pd.DataFrame()
    finally:
        print(f'{symbol} has been downloaded')

# def save