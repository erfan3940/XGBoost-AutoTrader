
import pandas as pd
import numpy as np

def backtest_signals_tp_sl(
    close_series: pd.Series,
    y_pred: pd.Series,
    tp_pct: float = 0.006,
    sl_pct: float = 0.002,
    hold_window: int = 25,
    wait_until_hit: bool = True,
    neutral_class: int = 1,
    long_class: int = 2,
    short_class: int = 0
):
    """
    Backtest using CLOSE-only logic exactly as requested.
    
    Parameters
    ----------
    close_series : pd.Series
        Close prices with DatetimeIndex (validation segment). 
    y_pred : pd.Series
        Model predictions aligned to the same index as `close_series`. Values in {0,1,2}.
    tp_pct : float
        Take profit as fraction of entry price (e.g., 0.006 for +0.6%).
    sl_pct : float
        Stop loss as fraction of entry price (e.g., 0.002 for -0.2%).
    hold_window : int
        Initial lookahead window (in bars). We check the next `hold_window` closes for TP/SL first-hit.
    wait_until_hit : bool
        If True and neither TP nor SL hits within `hold_window`, keep scanning forward until one is hit
        or until the end of data. If False, exit at the end of the window with no PnL (or optional mark-to-market).
    neutral_class, long_class, short_class : int
        Class mapping. Defaults to {0=short, 1=neutral, 2=long}.
    
    Returns
    -------
    trades : pd.DataFrame
        One row per executed trade with columns:
        ['entry_time','exit_time','direction','entry','exit','bars_held','hit','ret_pct']
    equity : pd.Series
        Cumulative equity curve assuming compounding: equity = cumprod(1 + ret_pct).
    stats : dict
        Simple statistics (win rate, avg return, total return, #trades, average bars, etc.).
    """
    # Ensure alignment & sorting
    close = close_series.dropna().astype(float).copy()
    y = y_pred.reindex(close.index).astype(int).copy()
    close = close.loc[y.index]  # align strictly
    idx = close.index

    trades = []
    in_trade = False
    direction = None
    entry_price = None
    entry_time = None

    i = 0
    n = len(idx)

    while i < n:
        t = idx[i]
        sig = y.iloc[i]

        if not in_trade:
            if sig == long_class or sig == short_class:
                in_trade = True
                direction = "long" if sig == long_class else "short"
                entry_price = float(close.iloc[i])  # entry at current close
                entry_time = t

                # Set target and stop levels
                if direction == "long":
                    tp_level = entry_price * (1.0 + tp_pct)
                    sl_level = entry_price * (1.0 - sl_pct)
                else:
                    tp_level = entry_price * (1.0 - tp_pct)  # price falling is profit
                    sl_level = entry_price * (1.0 + sl_pct)

                # Look ahead
                hit = None
                exit_price = None
                exit_time = None

                # First, scan the next `hold_window` closes for which hits first
                j = i + 1
                last_scan = min(n - 1, i + hold_window)
                while j <= last_scan:
                    c = float(close.iloc[j])
                    # "first hit wins"
                    if direction == "long":
                        if c <= sl_level:
                            hit, exit_price, exit_time = "SL", c, idx[j]
                            break
                        if c >= tp_level:
                            hit, exit_price, exit_time = "TP", c, idx[j]
                            break
                    else:  # short
                        if c >= sl_level:
                            hit, exit_price, exit_time = "SL", c, idx[j]
                            break
                        if c <= tp_level:
                            hit, exit_price, exit_time = "TP", c, idx[j]
                            break
                    j += 1

                # If neither hit in initial window and we choose to wait, continue scanning
                if hit is None and wait_until_hit:
                    while j < n:
                        c = float(close.iloc[j])
                        if direction == "long":
                            if c <= sl_level:
                                hit, exit_price, exit_time = "SL", c, idx[j]
                                break
                            if c >= tp_level:
                                hit, exit_price, exit_time = "TP", c, idx[j]
                                break
                        else:
                            if c >= sl_level:
                                hit, exit_price, exit_time = "SL", c, idx[j]
                                break
                            if c <= tp_level:
                                hit, exit_price, exit_time = "TP", c, idx[j]
                                break
                        j += 1

                # If still no hit and we don't wait, exit at window end with no PnL (or mark-to-market if desired)
                if hit is None and not wait_until_hit:
                    exit_time = idx[last_scan]
                    exit_price = float(close.iloc[last_scan])
                    hit = "NONE"

                # If still no hit and waited until end-of-data, close at last bar
                if hit is None and wait_until_hit:
                    exit_time = idx[-1]
                    exit_price = float(close.iloc[-1])
                    hit = "EOD"

                # Compute return in %
                if direction == "long":
                    ret_pct = (exit_price / entry_price) - 1.0
                else:
                    ret_pct = (entry_price / exit_price) - 1.0

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "direction": direction,
                    "entry": entry_price,
                    "exit": exit_price,
                    "bars_held": (close.index.get_loc(exit_time) - close.index.get_loc(entry_time)),
                    "hit": hit,
                    "ret_pct": ret_pct
                })

                # Move i forward to the bar AFTER exit (ignore intermediate signals while in trade)
                i = close.index.get_loc(exit_time) + 1
                
                # Every signal to open a trade                                                                  # editable
                #i+=1 
                
                in_trade = False
                direction = None
                entry_price = None
                entry_time = None
                continue  # continue main loop from updated i

        # If no trade opened, just advance
        i += 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        equity = pd.Series([1.0], index=[close.index[0]])
        stats = {"trades": 0, "win_rate": np.nan, "total_return_pct": 0.0}
        return trades_df, equity, stats

    equity = (1.0 + trades_df["ret_pct"]).cumprod()
    equity.index = trades_df["exit_time"]

    wins = (trades_df["ret_pct"] > 0).sum()
    stats = {
        "trades": len(trades_df),
        "win_rate": wins / len(trades_df),
        "avg_ret_pct": trades_df["ret_pct"].mean(),
        "median_ret_pct": trades_df["ret_pct"].median(),
        "total_return_pct": equity.iloc[-1] - 1.0,
        "avg_bars": trades_df["bars_held"].mean(),
        "tp_hits": int((trades_df["hit"] == "TP").sum()),
        "sl_hits": int((trades_df["hit"] == "SL").sum()),
        "none_hits": int((trades_df["hit"] == "NONE").sum()),
        "eod_closes": int((trades_df["hit"] == "EOD").sum())
    }
    return trades_df, equity, stats



# real backtest with initial and lot size
def backtest_signals_tp_sl_multi(
    close_series: pd.Series,
    y_pred: pd.Series,
    tp_pct: float = 0.006,
    sl_pct: float = 0.002,
    hold_window: int = 25,
    wait_until_hit: bool = True,
    neutral_class: int = 1,
    long_class: int = 2,
    short_class: int = 0,
    max_positions: int = 3,
    initial_balance: float = 1000.0,
    lot_size: float = 1.0
):
    """
    Backtest allowing multiple open trades (up to max_positions).
    Each signal can trigger a trade if positions < max_positions.

    Parameters
    ----------
    initial_balance : float
        Starting account balance in USD.
    lot_size : float
        Position size multiplier (PnL = ret_pct * lot_size * initial_balance_per_trade).
    """

    close = close_series.dropna().astype(float).copy()
    y = y_pred.reindex(close.index).astype(int).copy()
    idx = close.index
    n = len(idx)

    trades = []
    active_trades = []

    balance_curve = [initial_balance]
    balance_times = [idx[0]]
    balance = initial_balance

    for i in range(n):
        t = idx[i]
        price = float(close.iloc[i])
        sig = y.iloc[i]

        # --- Check if any active trade hits TP/SL ---
        still_open = []
        for trade in active_trades:
            direction = trade["direction"]
            entry_price = trade["entry"]
            tp_level = trade["tp"]
            sl_level = trade["sl"]

            hit = None
            if direction == "long":
                if price <= sl_level:
                    hit = "SL"
                elif price >= tp_level:
                    hit = "TP"
            else:  # short
                if price >= sl_level:
                    hit = "SL"
                elif price <= tp_level:
                    hit = "TP"

            if hit is not None:
                # Return in % (relative to entry)
                ret_pct = (price / entry_price - 1.0) if direction == "long" else (entry_price / price - 1.0)
                pnl = balance * (lot_size * ret_pct / max_positions)  # scale PnL per trade

                balance += pnl
                balance_curve.append(balance)
                balance_times.append(t)

                trades.append({
                    "entry_time": trade["entry_time"],
                    "exit_time": t,
                    "direction": direction,
                    "entry": entry_price,
                    "exit": price,
                    "bars_held": i - trade["i_entry"],
                    "hit": hit,
                    "ret_pct": ret_pct,
                    "pnl_usd": pnl,
                    "balance_after": balance
                })
            else:
                still_open.append(trade)

        active_trades = still_open

        # --- Open new trade if signal is long/short and positions < max ---
        if sig in [long_class, short_class] and len(active_trades) < max_positions:
            direction = "long" if sig == long_class else "short"
            entry_price = price
            tp_level = entry_price * (1 + tp_pct) if direction == "long" else entry_price * (1 - tp_pct)
            sl_level = entry_price * (1 - sl_pct) if direction == "long" else entry_price * (1 + sl_pct)

            active_trades.append({
                "direction": direction,
                "entry": entry_price,
                "entry_time": t,
                "tp": tp_level,
                "sl": sl_level,
                "i_entry": i
            })

    # --- Close remaining trades at end-of-data ---
    for trade in active_trades:
        price = float(close.iloc[-1])
        ret_pct = (price / trade["entry"] - 1.0) if trade["direction"] == "long" else (trade["entry"] / price - 1.0)
        pnl = balance * (lot_size * ret_pct / max_positions)
        balance += pnl
        balance_curve.append(balance)
        balance_times.append(idx[-1])

        trades.append({
            "entry_time": trade["entry_time"],
            "exit_time": idx[-1],
            "direction": trade["direction"],
            "entry": trade["entry"],
            "exit": price,
            "bars_held": n - trade["i_entry"] - 1,
            "hit": "EOD",
            "ret_pct": ret_pct,
            "pnl_usd": pnl,
            "balance_after": balance
        })

    trades_df = pd.DataFrame(trades)
    balance_series = pd.Series(balance_curve, index=balance_times)

    if trades_df.empty:
        stats = {"trades": 0, "win_rate": np.nan, "final_balance": initial_balance}
        return trades_df, balance_series, stats

    wins = (trades_df["ret_pct"] > 0).sum()
    stats = {
        "trades": len(trades_df),
        "win_rate": wins / len(trades_df),
        "avg_ret_pct": trades_df["ret_pct"].mean(),
        "median_ret_pct": trades_df["ret_pct"].median(),
        "final_balance": balance,
        "total_return_pct": (balance / initial_balance) - 1.0,
        "avg_bars": trades_df["bars_held"].mean(),
        "tp_hits": int((trades_df["hit"] == "TP").sum()),
        "sl_hits": int((trades_df["hit"] == "SL").sum()),
        "eod_closes": int((trades_df["hit"] == "EOD").sum())
    }

    return trades_df, balance_series, stats

def backtest_signals_tp_sl_multi_realistic(
    close_series: pd.Series,
    y_pred: pd.Series,
    tp_pct: float = 0.006,
    sl_pct: float = 0.002,
    hold_window: int = 25,
    wait_until_hit: bool = True,
    neutral_class: int = 1,
    long_class: int = 2,
    short_class: int = 0,
    max_positions: int = 3,
    initial_balance: float = 1000.0,
    risk_per_trade: float = 0.01,  # 1% of account per trade
    commission: float = 0.0005,    # 0.05% commission
    slippage: float = 0.0001       # 0.01% slippage
):
    """
    Realistic backtest with proper position sizing and transaction costs.
    """
    close = close_series.dropna().astype(float).copy()
    y = y_pred.reindex(close.index).astype(int).copy()
    idx = close.index
    n = len(idx)

    trades = []
    active_trades = []
    balance = initial_balance
    equity_curve = [balance]
    equity_times = [idx[0]]
    
    # For tracking maximum drawdown
    peak_balance = initial_balance
    max_drawdown = 0

    for i in range(n):
        t = idx[i]
        price = float(close.iloc[i])
        sig = y.iloc[i]

        # Process active trades
        still_open = []
        for trade in active_trades:
            direction = trade["direction"]
            entry_price = trade["entry"]
            tp_level = trade["tp"]
            sl_level = trade["sl"]
            position_size = trade["position_size"]

            hit = None
            exit_price = price
            
            # Check for TP/SL hits
            if direction == "long":
                if price <= sl_level:
                    hit = "SL"
                    exit_price = sl_level * (1 - slippage)  # Slippage on exit
                elif price >= tp_level:
                    hit = "TP"
                    exit_price = tp_level * (1 - slippage)
            else:  # short
                if price >= sl_level:
                    hit = "SL"
                    exit_price = sl_level * (1 + slippage)
                elif price <= tp_level:
                    hit = "TP"
                    exit_price = tp_level * (1 + slippage)

            if hit is not None:
                # Calculate returns with commissions
                if direction == "long":
                    gross_ret_pct = (exit_price / entry_price - 1.0)
                else:
                    gross_ret_pct = (entry_price / exit_price - 1.0)
                
                # Deduct commissions (entry + exit)
                net_ret_pct = gross_ret_pct - (2 * commission)
                pnl = position_size * net_ret_pct

                balance += pnl
                equity_curve.append(balance)
                equity_times.append(t)
                
                # Update drawdown tracking
                if balance > peak_balance:
                    peak_balance = balance
                current_drawdown = (peak_balance - balance) / peak_balance
                max_drawdown = max(max_drawdown, current_drawdown)

                trades.append({
                    "entry_time": trade["entry_time"],
                    "exit_time": t,
                    "direction": direction,
                    "entry": entry_price,
                    "exit": exit_price,
                    "bars_held": i - trade["i_entry"],
                    "hit": hit,
                    "ret_pct": net_ret_pct,
                    "pnl_usd": pnl,
                    "balance_after": balance,
                    "position_size": position_size
                })
            else:
                still_open.append(trade)

        active_trades = still_open

        # Open new trade if conditions met
        if (sig in [long_class, short_class] and 
            len(active_trades) < max_positions and
            balance > initial_balance * 0.5):  # Prevent trading with <50% capital
            
            direction = "long" if sig == long_class else "short"
            entry_price = price * (1 + slippage) if direction == "long" else price * (1 - slippage)
            
            # Calculate position size based on risk
            risk_amount = balance * risk_per_trade
            if direction == "long":
                risk_per_share = entry_price * sl_pct
            else:
                risk_per_share = entry_price * sl_pct
                
            position_size = min(risk_amount / risk_per_share, balance / entry_price)
            
            # Deduct entry commission
            commission_cost = position_size * entry_price * commission
            balance -= commission_cost
            
            tp_level = entry_price * (1 + tp_pct) if direction == "long" else entry_price * (1 - tp_pct)
            sl_level = entry_price * (1 - sl_pct) if direction == "long" else entry_price * (1 + sl_pct)

            active_trades.append({
                "direction": direction,
                "entry": entry_price,
                "entry_time": t,
                "tp": tp_level,
                "sl": sl_level,
                "i_entry": i,
                "position_size": position_size
            })

    # Close remaining trades
    for trade in active_trades:
        exit_price = float(close.iloc[-1]) * (1 - slippage) if trade["direction"] == "long" else float(close.iloc[-1]) * (1 + slippage)
        
        if trade["direction"] == "long":
            gross_ret_pct = (exit_price / trade["entry"] - 1.0)
        else:
            gross_ret_pct = (trade["entry"] / exit_price - 1.0)
        
        net_ret_pct = gross_ret_pct - (2 * commission)
        pnl = trade["position_size"] * net_ret_pct
        balance += pnl
        
        trades.append({
            "entry_time": trade["entry_time"],
            "exit_time": idx[-1],
            "direction": trade["direction"],
            "entry": trade["entry"],
            "exit": exit_price,
            "bars_held": n - trade["i_entry"] - 1,
            "hit": "EOD",
            "ret_pct": net_ret_pct,
            "pnl_usd": pnl,
            "balance_after": balance,
            "position_size": trade["position_size"]
        })

    equity_series = pd.Series(equity_curve, index=equity_times)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # Calculate statistics
    if trades_df.empty:
        stats = {
            "trades": 0, 
            "final_balance": initial_balance,
            "total_return_pct": 0.0,
            "max_drawdown": 0.0
        }
    else:
        wins = (trades_df["ret_pct"] > 0).sum()
        stats = {
            "trades": len(trades_df),
            "win_rate": wins / len(trades_df),
            "avg_ret_pct": trades_df["ret_pct"].mean(),
            "final_balance": balance,
            "total_return_pct": (balance / initial_balance) - 1.0,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": (trades_df["ret_pct"].mean() / trades_df["ret_pct"].std()) * np.sqrt(252) if len(trades_df) > 1 else 0,
            "profit_factor": abs(trades_df[trades_df["pnl_usd"] > 0]["pnl_usd"].sum() / 
                               trades_df[trades_df["pnl_usd"] < 0]["pnl_usd"].sum()) if trades_df[trades_df["pnl_usd"] < 0]["pnl_usd"].sum() != 0 else float('inf')
        }

    return trades_df, equity_series, stats


def backtest_signals_tp_sl_final(
    close_series: pd.Series,
    y_pred: pd.Series,
    tp_pct: float = 0.006,
    sl_pct: float = 0.002,
    hold_window: int = 25,
    wait_until_hit: bool = True,
    neutral_class: int = 1,
    long_class: int = 2,
    short_class: int = 0,
    max_positions: int = 3,
    initial_balance: float = 1000.0,
    lot_size: float = 1.0,   # lot = risk per trade multiplier
    compound: bool = True    # if False → balance stays fixed, pnl just adds/subtracts
):
    """
    Multi-position backtester with TP/SL and balance tracking.
    """
    close = close_series.dropna().astype(float).copy()
    y = y_pred.reindex(close.index).astype(int).copy()
    idx = close.index

    trades = []
    balance = initial_balance
    equity_curve = []

    open_positions = []  # list of dicts: {entry_time, direction, entry, tp, sl, bars_open}

    for i, t in enumerate(idx):
        price = float(close.iloc[i])
        sig = y.iloc[i]

        # --- update open positions ---
        still_open = []
        for pos in open_positions:
            pos["bars_open"] += 1
            exit_flag = None
            exit_price = None

            if pos["direction"] == "long":
                if price <= pos["sl"]:
                    exit_flag, exit_price = "SL", price
                elif price >= pos["tp"]:
                    exit_flag, exit_price = "TP", price
            else:  # short
                if price >= pos["sl"]:
                    exit_flag, exit_price = "SL", price
                elif price <= pos["tp"]:
                    exit_flag, exit_price = "TP", price

            if exit_flag is None:
                if not wait_until_hit and pos["bars_open"] >= hold_window:
                    exit_flag, exit_price = "TIME", price
                elif wait_until_hit and i == len(idx) - 1:  # last bar
                    exit_flag, exit_price = "EOD", price

            if exit_flag:
                ret_pct = (exit_price / pos["entry"] - 1.0) if pos["direction"] == "long" else (pos["entry"] / exit_price - 1.0)
                trade_size = balance * lot_size if compound else initial_balance * lot_size
                pnl_usd = trade_size * ret_pct
                balance += pnl_usd

                trades.append({
                    "entry_time": pos["entry_time"],
                    "exit_time": t,
                    "direction": pos["direction"],
                    "entry": pos["entry"],
                    "exit": exit_price,
                    "bars_held": pos["bars_open"],
                    "hit": exit_flag,
                    "ret_pct": ret_pct,
                    "pnl_usd": pnl_usd,
                    "balance": balance
                })
            else:
                still_open.append(pos)

        open_positions = still_open
        equity_curve.append((t, balance))

        # --- open new position if signal ---
        if sig in [long_class, short_class] and len(open_positions) < max_positions:
            direction = "long" if sig == long_class else "short"
            entry_price = price
            tp_level = entry_price * (1 + tp_pct) if direction == "long" else entry_price * (1 - tp_pct)
            sl_level = entry_price * (1 - sl_pct) if direction == "long" else entry_price * (1 + sl_pct)

            open_positions.append({
                "entry_time": t,
                "direction": direction,
                "entry": entry_price,
                "tp": tp_level,
                "sl": sl_level,
                "bars_open": 0
            })

    trades_df = pd.DataFrame(trades)
    equity = pd.Series({t: b for t, b in equity_curve})

    if trades_df.empty:
        return trades_df, equity, {"trades": 0}

    stats = {
        "trades": len(trades_df),
        "win_rate": (trades_df["ret_pct"] > 0).mean(),
        "avg_ret_pct": trades_df["ret_pct"].mean(),
        "median_ret_pct": trades_df["ret_pct"].median(),
        "final_balance": balance,
        "total_return_pct": balance / initial_balance - 1.0,
        "avg_bars": trades_df["bars_held"].mean(),
        "tp_hits": int((trades_df["hit"] == "TP").sum()),
        "sl_hits": int((trades_df["hit"] == "SL").sum()),
        "eod_closes": int((trades_df["hit"] == "EOD").sum()),
    }

    return trades_df, equity, stats
