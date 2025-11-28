import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict
import importlib
import time
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
import config as cfg
import data_manager as dm
import MT5_connector as mc
import MetaTrader5 as mt5
import feature_generator as fg
import order_manager as om
import model_manager as mm
config = cfg.get_config()
# --------------------------- Logging ---------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("main")
# --------------------------- Performance ------------------------------
def ensure_perf_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=cfg.PERF_COLUMNS)
    # ensure columns exist even if loaded file is older version
    for c in cfg.PERF_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[cfg.PERF_COLUMNS]
# --------------------------- Core Steps -------------------------------

def update_base_dataframe(symbol: str, base_df: pd.DataFrame) -> pd.DataFrame:
    """Fetch the recent bar and append it """
    global bars_counter
    last_bar = dm.get_mt5_rates(symbol, mt5.TIMEFRAME_H1,"time_range",from_time = base_df.index.max(), to_time = datetime.now() + timedelta(hours=3))
    if last_bar is None or len(last_bar)==0:
        logger.warning("No last bar received from MT5; leaving DF unchanged.")
        return base_df
    # check last bar
    # Filter only new bars that don't exist in base_df
    new_bars = last_bar[~last_bar.index.isin(base_df.index)]
    bars_counter += len(new_bars)
    if len(new_bars) == 0:
        logger.info("No new bars to append")
        return base_df
    # Append all new bars at once
    base_df = pd.concat([base_df, new_bars[["Open", "High", "Low", "Close", "Volume"]]])
    base_df = base_df.sort_index()
    
    logger.info("Appended %d new bars", len(new_bars))
    base_df.to_parquet(config["base_df_path"])
    return base_df

def trade_restricts(symbol: str, config: dict) -> bool:
    open_positions = mt5.positions_get()
    n_open = len(open_positions)
    
    if (config["wait_until_flat"]== True) and n_open > 0: 
        logger.info("wait_until_flat=True and %d open positions -> skip trading.", n_open)
        return False

    if n_open >= config["max_open_positions"] :
        logger.info("Max open positions reached (%d) -> skip trading.", n_open)
        return False

    return True
# ---------------------- Performance Updating --------------------------

def update_performance(symbol: str, perf_df: pd.DataFrame, since_ts: pd.Timestamp) -> pd.DataFrame:
    
    closed = mt5.history_deals_get(since_ts, datetime.now()+timedelta(hours=3))  # since_ts
    if not closed:
        print("nothing in performace")
        return perf_df

    rows = []
    for deal in closed:
        deal[0]
        ticket = deal.ticket                         
        direction = "buy" if deal.type in ("buy", 0) else "sell" # 0= buy 0=in
        entry_price = float(deal.price)
        commission = float(deal.commission)
        pnl = float(deal.profit)
        comment = deal.comment
        conf = _parse_confidence_from_comment(comment)

        rows.append({
            "ticket": ticket,
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "commission" : commission,
            "pnl": pnl,
            "comment": comment,
        })

    if rows:
        print("i added some rows")
        perf_df = pd.concat([perf_df, pd.DataFrame(rows)], ignore_index=True)
        perf_df = perf_df.drop_duplicates(subset=["ticket"], keep="last")
        perf_df = perf_df.sort_values("ticket")
    return perf_df


def _parse_confidence_from_comment(comment: str) -> Optional[float]:
    try:
        # comment format e.g., "ML:buy|conf:0.62"
        if "conf:" in comment:
            return float(comment.split("conf:")[-1].split(" ")[0].split("|")[0])
    except Exception:
        pass
    return None


def _bars_between(start: pd.Timestamp, end: pd.Timestamp, bar_hours: int = 1) -> int:
    if pd.isna(start) or pd.isna(end):
        return None
    delta = end - start
    #return int(round(delta.total_seconds() / 3600 / bar_hours))
    return int(round(delta.total_seconds() / 60 ))


def winrate_calc(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    df['balance'] = df['pnl'].cumsum()
    df['winrate'] = np.nan
    for i in range(1, len(df)):
        subset = df.iloc[1:i+1]
        positive_count = (subset['pnl'] > 0).sum()     
        total_trades = len(subset)
        win_rate = (positive_count / total_trades) * 100 if total_trades > 0 else 0
        df.loc[df.index[i],'winrate'] = win_rate
    print(df.tail(3))
    return df
# --------------------------- Main ------------------------------------
bars_counter = 0
def main():
    global bars_counter
    config = cfg.get_config()
    mc.initialize_MT5()
    symbol = config["symbol"]

    # Load dataframes
    base_df = pd.read_parquet(config["base_df_path"])
    perf_df = pd.read_parquet(config["performance_log_path"])
    perf_df = ensure_perf_schema(perf_df)

    last_bar_time = base_df.index.max() if not base_df.empty else None

    while True:
        try:
            print("Running trading cycle...")
            print(f"bars_counter = {bars_counter}")
            time.sleep(60)

            # ---------------- Fetch new bar ----------------
            new_base_df = update_base_dataframe(symbol, base_df)

            # if no new closed bar → skip this cycle
            if new_base_df.index.max() == last_bar_time:
                logger.info("No new closed bar -> skip prediction")
                continue

            # update state
            base_df = new_base_df
            last_bar_time = base_df.index.max()
            
            # ---------------- Feature generation ----------------
            base = fg.MT5FeaturesManager(base_df)
            base_df = base.add_all_features()
            base_df = base.handle_missing_values('drop')

            last_row = base_df.tail(1)
            if last_row.empty:
                print("No features available yet -> skip this cycle")
                continue

            feats = last_row[cfg.MODEL2_FEATURES_H1]
            if feats.empty:
                print("Features dataframe empty -> skip this cycle")
                continue

            # ---------------- Prediction ----------------
            pred = mm.make_prediction(config['model_path_M15'], feats, config)
            if pred is not None:
                print(f"pred: {pred[0]} conf: {pred[1]:.2f}")
                can_trade = trade_restricts(symbol, config)
            else:
                print("prediction is None or below threshold")
                can_trade = False

            # ---------------- Trade execution ----------------
            if can_trade:
                entry_price = float(last_row.iloc[-1]['Close'])
                trade = om.send_order(entry_price, direction=pred[0], confidence=pred[1], config=config)
                logger.info("Trade result: %s", trade)
            else:
                logger.info("No trade placed (trade restricts or None prediction).")

            # ---------------- Performance update ----------------
            last_exit = (
                pd.to_datetime(perf_df["ticket"].max(), utc=True)
                if not perf_df.empty
                else pd.Timestamp.utcnow() - pd.Timedelta(days=7)
            )
            perf_df = update_performance(symbol, perf_df, last_exit)

            # compute rolling winrate
            perf_df = winrate_calc(perf_df)
            perf_df.to_parquet(config["performance_log_path"])

            last_winrate = perf_df.iloc[-1]['winrate']
            if last_winrate is not None:
                logger.info("total winrate_calc (last %s): %.2f", len(perf_df), last_winrate)
                if last_winrate < config["stop_cutoff"]:
                    logger.warning("Winrate %.2f below cutoff : %.2f -> STOP TRADING.",
                                   last_winrate, config['stop_cutoff'])
                    mt5.shutdown()
                    break

            # ---------------- Retraining ----------------
            if bars_counter % 2 == 0:  # adjust as needed
                print(f"{bars_counter / 2} steps passed -> retrain model")
                mm.retrain_model(mt5.TIMEFRAME_H1, config)

        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(5)
            continue


if __name__ == "__main__":
    main()