# XGBoost-AutoTrader
 # 📈 XGB AutoTrader  
Automated ML Trading System using XGBoost + Walk-Forward Training for MetaTrader 5

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()  
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)]()  
[![Status](https://img.shields.io/badge/Build-Stable-brightgreen.svg)]()

---

## 🔍 Overview  
**XGB AutoTrader** is an automated machine-learning trading system built with **XGBoost**, designed to forecast market direction on **EURUSD** and **XAUUSD** using **walk-forward training**, dynamic retraining cycles, and live execution through **MetaTrader 5**.

The bot slices historical data into multiple segments, trains on each slice (WFA), and retrains automatically at configurable intervals depending on your timeframe (M5, M15, H1, H4).  
It includes full logging, risk management, position sizing, and both static & dynamic backtesting.

---

# 🚀 Features  
### Core Capabilities
- ✔ XGBoost ML model for classification/forecasting  
- ✔ Walk-Forward Analysis (WFA) with automatic model retraining  
- ✔ MetaTrader 5 live price feed (real-time OHLCV)  
- ✔ Risk & money management  
- ✔ Position sizing engine (lot calculation per trade)  
- ✔ Stop Loss & Take Profit logic  
- ✔ Max simultaneous position control  
- ✔ Pause trading if win rate < 45%  
- ✔ Full terminal logs (signals, confidence, retrain events)  

### Backtesting
- ✔ Static backtest (single-period)  
- ✔ Walk-forward backtest (dynamic)  

### Performance Tracking
- ✔ Records model confidence  
- ✔ Tracks SL/TP hit rate  
- ✔ Logs retrain cycles  
- ✔ Saves model for next sessions  

---

# 🧠 Market & Timeframe Support  
The bot is built for any MetaTrader 5 symbol, but tested mainly on:

| Symbol | Status | Notes |
|--------|--------|-------|
| **EURUSD** | ⭐ Best results | Best precision in WFA testing |
| **XAUUSD** | Good | Volatile, requires careful risk management |

### Timeframes Tested:
- M5  
- M15  
- **H1 (Best results, optimal for feature engineering)**  
- H4  

---

# 🧬 Architecture Overview  
/XGB_AutoTrader
│── src/
│ ├── main.py # main trading loop
│ ├── order_manager.py # send orders
│ ├── backtest_tp_sl_window.py # include different types of backtest
│ ├── MT5_connetor.py # essensials for mt5 conneting
│ ├── model_maker_static.ipynb # static backtest
│ ├── model_maker_dynamic.ipynb # WFA backtest
│ ├── data_manager.py # get and control data flow
│ ├── feature_generator.py # almost all usefull features for trading
│ ├── config.py (NOT INCLUDED - PRIVATE) # has selected models features
│── LICENSE (Apache 2.0)
│── README.md
├── walkforwards # backtests resualts in different symbols, timeframe and feartures and ...
├── models # last retrained models
├── data # price feed for training
├── model_logs # model saved logs

# 📉 Real-World VPS Testing

This bot has been running live for 2 months on a VPS trading EURUSD.
During development, debugging occurred at times, so real results are not 100% clean — but backtest precision is above 50%, and walk-forward results show promising robustness.

# 📊 Performance (Screenshots)
<img width="1082" height="831" alt="Screenshot 2025-11-28 215525" src="https://github.com/user-attachments/assets/18406bef-aee9-404c-b43a-d41709de96fc" />
<img width="1080" height="856" alt="Screenshot 2025-11-28 215612" src="https://github.com/user-attachments/assets/4d275f7a-b4c6-4695-98f9-3325e9ddb5a1" />

more statistics in Trade Report pdf file [Trade report-678086 2025-11-14 14-40.pdf](https://github.com/user-attachments/files/23828791/Trade.report-678086.2025-11-14.14-40.pdf)

This project is licensed under the Apache License 2.0.
See the LICENSE file for full details.

If you want, I can expand this bot to be more profitable
Just tell me!
# 🔧 Installation & Setup  

### 1. Clone Repository
```bash
git clone https://github.com/yourname/XGB-AutoTrader.git
cd XGB-AutoTrader
