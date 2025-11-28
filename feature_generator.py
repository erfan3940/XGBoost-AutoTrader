import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
import ta
from ta.trend import ADXIndicator
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import linregress
import warnings
warnings.filterwarnings('default') # not ignore

class MT5FeaturesManager:
    def __init__(self, df: pd.DataFrame = None):
        """
        Initialize Features Manager with MT5 DataFrame
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data from MT5
        """
        self.df = df
        self.feature_groups = {}
        self.scalers = {}
        
    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Set or update the DataFrame"""
        self.df = df.copy()
        
    def validate_dataframe(self) -> bool:
        """Validate if DataFrame has required columns"""
        if self.df is None:
            raise ValueError("DataFrame doesnt exist. check set_dataframe() first.")
        
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return True
    
    def add_basic_features(self) -> pd.DataFrame:
        
        self.validate_dataframe()
        
        df = self.df.copy()
        
        # Price changes
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)  # Annualized bcz std() and variance(std^2) scales linearly with time
        
        # High-Low features
        df['HL_Ratio'] = df['High'] / df['Low']
        df['OC_Ratio'] = df['Open'] / df['Close']
        
        # Price positions
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, 1e-10)
        
        self.df = df
        self.feature_groups['basic'] = ['Returns', 'Log_Returns', 'Volatility', 
                                       'HL_Ratio', 'OC_Ratio', 'Price_Position']
        return df
    
    def add_technical_indicators(self, 
                               ma_windows: List[int] = [10, 20, 50, 100, 200],
                               rsi_window: List[int] = [14, 21],
                               atr_window: List[int] = [14, 21],
                               bb_window: int = 20,
                               adx_window: int = 14,
                               last_RP: int = 100,        
                               fibo_window: int = 70,
                               stoch_window: int = 14) -> pd.DataFrame:
        
        self.validate_dataframe()
        
        df = self.df.copy()
        
        # Moving Averages
        for window in ma_windows:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'MA_Ratio_{window}'] = df['Close'] / df[f'MA_{window}']
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df["MA_Cross50/100"] = np.where(df["MA_50"]>df["MA_100"],1,-1)
        df["MA_Cross100/200"] = np.where(df["MA_100"]>df["MA_200"],1,-1)
        df["GD_Cross"] = np.where(df["MA_50"]>df["MA_200"],1,-1)

        # Distance to MA
        df['dist_to_MA_200'] = (df['Close']-df['MA_200'])/df['Close']
        df['dist_to_MA_100'] = (df['Close']-df['MA_100'])/df['Close']
        df['dist_to_MA_50'] = (df['Close']-df['MA_50'])/df['Close']
        df['dist_to_MA_20'] = (df['Close']-df['MA_20'])/df['Close']
        # RSI
        for window in rsi_window:
            df[f"RSI_{window}"] = ta.momentum.rsi(df["Close"], window=window)

        #ATR
        for window in atr_window:
            df[f"ATR_{window}"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=window)
        
        # Bollinger Bands
        indicator_bb = ta.volatility.BollingerBands(df['Close'],window=bb_window, window_dev=2)
        df['BB_High'] = indicator_bb.bollinger_hband()
        df['BB_Low'] = indicator_bb.bollinger_lband()
        df['BB_Mid'] = indicator_bb.bollinger_mavg()
        df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid']

        # Calculate recent high and low
        df['Recent_High'] = df['High'].rolling(window=fibo_window).max()
        df['Recent_Low'] = df['Low'].rolling(window=fibo_window).min()
        
        # Fibonacci levels
        df['Fib_236'] = df['Recent_High'] - 0.236 * (df['Recent_High'] - df['Recent_Low'])
        df['Fib_382'] = df['Recent_High'] - 0.382 * (df['Recent_High'] - df['Recent_Low'])
        df['Fib_500'] = df['Recent_High'] - 0.5 * (df['Recent_High'] - df['Recent_Low'])
        df['Fib_618'] = df['Recent_High'] - 0.618 * (df['Recent_High'] - df['Recent_Low'])
        df['Fib_786'] = df['Recent_High'] - 0.786 * (df['Recent_High'] - df['Recent_Low'])
        df['dist_to_fibo38'] = (df['Close'] - df['Fib_382']) / df['Close']
        df['dist_to_fibo50'] = (df['Close'] - df['Fib_500']) / df['Close']
        # Recent support (lows)
        df['Support_1'] = df['Low'].rolling(window=last_RP).min()
        df['Support_2'] = df['Low'].rolling(window=last_RP*2).min()
        df['dist_to_Support_1'] = (df['Close']-df['Support_1'])/df['Close'] 
        df['dist_to_Support_2'] = (df['Close']-df['Support_2'])/df['Close'] 
        
        
        # Recent resistance (highs)
        df['Resistance_1'] = df['High'].rolling(window=last_RP).max()
        df['Resistance_2'] = df['High'].rolling(window=last_RP*2).max()
        df['dist_to_Resistance_1'] = (df['Resistance_1']-df['Close'])/df['Close'] 
        df['dist_to_Resistance_2'] = (df['Resistance_2']-df['Close'])/df['Close'] 
        
        # Calculate Pivot Point
        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        
        # Support and Resistance Levels
        df['R1'] = (2 * df['Pivot']) - df['Low']
        df['S1'] = (2 * df['Pivot']) - df['High']
        
        df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
        df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
        
        df['R3'] = df['High'] + 2 * (df['Pivot'] - df['Low'])
        df['S3'] = df['Low'] - 2 * (df['High'] - df['Pivot'])
        #ADX
        adx_indicator = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=adx_window)   
        df['adx_plus_di'] = adx_indicator.adx_pos()  # +DI (positive directional indicator)
        df['adx_minus_di'] = adx_indicator.adx_neg()  # -DI (negative directional indicator)   
        df['adx_di_crossover'] = (df['adx_plus_di'] > df['adx_minus_di']).astype(int)  # 1 when +DI crosses above -DI
        df['adx_di_crossunder'] = (df['adx_plus_di'] < df['adx_minus_di']).astype(int)  # 1 when +DI crosses bellow -DI
            
        # Stochastic Oscillator
        low_min = df['Low'].rolling(window=stoch_window).min()
        high_max = df['High'].rolling(window=stoch_window).max()
        df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min).replace(0, 1e-10)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        self.df = df
        self.feature_groups['technical'] = [col for col in df.columns if col not in self.feature_groups.get('basic', [])]

        


        def trendline_distance(df: pd.DataFrame, lookback: int = 50, mode: str = 'support') -> pd.Series:
            """
            mode: 'support' (swing lows) or 'resistance' (swing highs)
            """
            distances = []
            for i in range(len(df)):
                if i < lookback:
                    distances.append(np.nan)
                    continue
                
                # Extract window
                window = df.iloc[i - lookback:i]
                if mode == 'support':
                    y = window['Low'].values
                else:
                    y = window['High'].values
                
                x = np.arange(len(y))
                slope, intercept, _, _, _ = linregress(x, y)
                
                # Predicted trendline value at current index
                current_x = len(y)
                trend_value = slope * current_x + intercept
                
                close_price = df.iloc[i]['Close']
                distances.append(close_price - trend_value)
            
            return pd.Series(distances, index=df.index, name=f"trend_dist_{mode}")
            
        df['trend_dist_support'] = trendline_distance(df, 200, 'support')
        df['trend_dist_resistance'] = trendline_distance(df, 200, 'resistance')
        
        # Normalize with ATR or Close price
        df['trend_dist_support_norm'] = df['trend_dist_support'] / df['Close']
        df['trend_dist_resistance_norm'] = df['trend_dist_resistance'] / df['Close']

        return df
    
    def add_volume_features(self) -> pd.DataFrame:
        
        self.validate_dataframe()
        
        if 'Volume' not in self.df.columns:
            print("Warning: Volume data not available")
            return self.df
        
        df = self.df.copy()
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA'].replace(0, 1e-10)
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Volume-price relationship
        df['Volume_Price_Correlation'] = df['Volume'].rolling(window=20).corr(df['Close'])
        
        # OBV (On Balance Volume)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        self.df = df
        self.feature_groups['volume'] = ['Volume_MA', 'Volume_Ratio', 'Volume_Change', 
                                       'Volume_Price_Correlation', 'OBV']
        return df
    
    def add_time_features(self) -> pd.DataFrame:
        
        self.validate_dataframe()
        
        df = self.df.copy()
        
        # Time features
        df['Hour'] = df.index.hour
        df['DayOfWeek'] = df.index.dayofweek
        df['DayOfMonth'] = df.index.day
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['year'] = df.index.year

        # Weekly support/resistance
        df['week_number'] = df.index.isocalendar().week
        df['weekly_high'] = df.groupby(['year', 'week_number'])['High'].transform('max')
        df['weekly_low'] = df.groupby(['year','week_number'])['Low'].transform('min')
        df['dist_to_weekly_high'] = (df['weekly_high']-df['Close'])/df['Close'] 
        df['dist_to_weekly_low'] = (df['Close']-df['weekly_low'])/df['Close']
        df['near_weekly_high'] = (df['dist_to_weekly_high']<0.008).astype(int)
        df['near_weekly_low'] = (df['dist_to_weekly_low']<0.008).astype(int)
        
        # Cyclical encoding for time features
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        
        # Market session flags
        df['Asian_Session'] = ((df['Hour'] >= 0) & (df['Hour'] < 8)).astype(int)
        df['European_Session'] = ((df['Hour'] >= 8) & (df['Hour'] < 16)).astype(int)
        df['US_Session'] = ((df['Hour'] >= 16) & (df['Hour'] < 24)).astype(int)
        
        self.df = df
        self.feature_groups['time'] = ['Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Quarter','year',
                                       'week_number', 'weekly_high', 'weekly_low', 'dist_to_weekly_high',
                                       'dist_to_weekly_low', 'near_weekly_high', 'near_weekly_low',
                                     'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
                                     'Asian_Session', 'European_Session', 'US_Session']
        return df
    
    def add_statistical_features(self, window: int = 20) -> pd.DataFrame:
        
        self.validate_dataframe()
        
        df = self.df.copy()
        
        # Statistical features
        df['Returns_Std'] = df['Returns'].rolling(window=window).std()
        df['Returns_Skew'] = df['Returns'].rolling(window=window).skew()
        df['Returns_Kurtosis'] = df['Returns'].rolling(window=window).kurt()
        
        # Z-score of returns
        returns_mean = df['Returns'].rolling(window=window).mean()
        returns_std = df['Returns'].rolling(window=window).std()
        df['Returns_ZScore'] = (df['Returns'] - returns_mean) / returns_std.replace(0, 1e-10)
        
        # Rolling correlations (if multiple symbols available)
        if 'Volume' in df.columns:
            df['Price_Volume_Corr'] = df['Close'].rolling(window=window).corr(df['Volume'])

        # Download VIX 
        vix = yf.download("^VIX", start=df.index.min(), end=df.index.max(), interval="1d")
        
        vix.columns = vix.columns.droplevel(1) # Drop the "Ticker" level from columns

        vix = vix.rename(columns={
            'Close': 'VIX_Close',
            'High': 'VIX_High',
            'Low': 'VIX_Low',
            'Open': 'VIX_Open',
            'Volume': 'VIX_Volume'
        })
        vix = vix[['VIX_Close']]

        vix.index = pd.to_datetime(vix.index)
        vix_hourly = vix.resample('H').ffill()
        df = df.merge(vix_hourly, how='left', left_index=True, right_index=True)
        self.df = df
        self.feature_groups['statistical'] = ['Returns_Std', 'Returns_Skew', 'Returns_Kurtosis',
                                            'Returns_ZScore', 'Price_Volume_Corr']
        return df
    
    def add_all_features(self, **kwargs) -> pd.DataFrame:
        
        df = self.df.copy()
        
        df = self.add_basic_features()
        df = self.add_technical_indicators(**kwargs)
        
        if 'Volume' in df.columns:
            df = self.add_volume_features()
        
        df = self.add_time_features()
        df = self.add_statistical_features()
        if df.isna().sum().sum() > 0:
            print("we found empty data after add all features")
            self.handle_missing_values(method='drop')
        self.df = df
        return df
    
    def handle_missing_values(self, method: str = 'ffill') -> pd.DataFrame:
        """
        'ffill & bfill'
        'drop'
        'interpolate'
        """
        if self.df is None:
            raise ValueError("DataFrame is None. get features before droping first.")
        self.validate_dataframe()
        
        df = self.df.copy()
        
        if method == 'drop': # accuracy is important
            df = df.dropna()
        elif method == 'ffill': # short gaps its commonly
            df = df.ffill().bfill()  # Forward fill then backward fill
        elif method == 'interpolate': # 1-2 bars but it invent data
            df = df.interpolate(method='linear')
        else:
            raise ValueError("Method must be 'drop', 'ffill', or 'interpolate'")
        
        self.df = df
        return df
    
    def scale_features(self, feature_list: List[str] = None, scaler_type: str = 'standard') -> pd.DataFrame:
        
        self.validate_dataframe()
        
        df = self.df.copy()
        
        if feature_list is None:
            # Scale all non-price and non-time features
            exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Hour', 'DayOfWeek', 
                          'DayOfMonth', 'Month', 'Quarter','year','week_number', 'Asian_Session', 'European_Session', 'US_Session']
            feature_list = [col for col in df.columns if col not in exclude_cols]
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        
        # Scale features
        scaled_features = scaler.fit_transform(df[feature_list])
        df_scaled = pd.DataFrame(scaled_features, index=df.index, columns=[f'{col}_scaled' for col in feature_list])
        
        # Store scaler for potential inverse transformation
        self.scalers[scaler_type] = scaler
        
        # Join scaled features with original DataFrame
        df = pd.concat([df, df_scaled], axis=1)
        
        self.df = df
        return df
    
    def get_feature_list(self, group: str = None) -> List[str]:
        """
        Get list of features by group or all features
        """
        if group:
            return self.feature_groups.get(group, [])
        else:
            return [feature for features in self.feature_groups.values() for feature in features]
    
    def get_feature_matrix(self, features: List[str] = None, drop_na: bool = True) -> pd.DataFrame:
        
        self.validate_dataframe()
        
        if features is None:
            features = self.get_feature_list()
        
        feature_matrix = self.df[features].copy()
        
        if drop_na:
            feature_matrix = feature_matrix.dropna()
        
        return feature_matrix
    
    def save_features(self, filepath: str, format: str = 'parquet') -> None:
        
        self.validate_dataframe()
        
        if format == 'parquet':
            self.df.to_parquet(filepath)
        elif format == 'csv':
            self.df.to_csv(filepath)
        elif format == 'feather':
            self.df.to_feather(filepath)
        else:
            raise ValueError("Format must be 'parquet', 'csv', or 'feather'")
    
    def load_features(self, filepath: str, format: str = 'parquet') -> pd.DataFrame:
        
        if format == 'parquet':
            df = pd.read_parquet(filepath)
        elif format == 'csv':
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        elif format == 'feather':
            df = pd.read_feather(filepath)
        else:
            raise ValueError("Format must be 'parquet', 'csv', or 'feather'")
        
        self.df = df
        return df


# Utility functions
def create_target_variable(df: pd.DataFrame, horizon: int = 1, method: str = 'dynamic',threash_num=1.1) -> pd.DataFrame:
    """
    dynamic or fixed
    """
    df = df.copy()
      # 0:short 1:range 2:long
    if method == 'dynamic':
        # based on ATR/volatility 
        fwd_ret = np.log(df["Close"]).shift(-horizon) - np.log(df["Close"])
        thr = threash_num * df["ATR_14"] / df["Close"]
        df["label"] = np.where(fwd_ret > thr, 2, np.where(fwd_ret < -thr, 0, 1))
    
    if method == 'fixed':
        # base on static 1% change
        fwd_ret = df['Close'].pct_change(horizon).shift(-horizon)
        thr = 0.005
        df["label"] = np.where(fwd_ret > thr, 2, np.where(fwd_ret < -thr, 0, 1))
    
    elif method == 'dynamic_max_min':
        labels = []
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        n = len(df)

        for i in range(n):
            entry = closes[i]
            label = 1  # default: range
            for j in range(1, horizon+1):
                if i+j >= n:  # avoid going past dataset
                    break
                # relative moves
                up_move = (highs[i+j] - entry) / entry
                down_move = (lows[i+j] - entry) / entry

                if up_move >= 0.005:
                    label = 2
                    break
                elif down_move <= -0.005:
                    label = 0
                    break
            labels.append(label)
        
        df['label'] = labels

    return df


def split_train_test(df: pd.DataFrame, test_size: float = 0.2, time_based: bool = True):
    
    if time_based:
        # Time-based split (chronological)
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
    else:
        # Random split
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=test_size, shuffle=False)
    
    return train_df, test_df