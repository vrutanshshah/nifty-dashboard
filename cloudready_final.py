import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from scipy.stats import linregress
from datetime import datetime, timedelta
import pytz
import requests
import time
import sqlite3

# ==========================================
# 1. Configuration & State
# ==========================================
st.set_page_config(page_title="Live Nifty Options Algo", layout="wide", page_icon="⚡")
IST = pytz.timezone('Asia/Kolkata')

class AlgoConfig:
    STRIKE_OFFSET = 100 
    SL_PCT = 0.20 
    TAKE_PROFIT_PCT = 0.40      
    TRAILING_SL_PCT = 0.10      
    LOT_SIZE = 65
    NUM_LOTS = 3                # NEW: Set how many lots you want to trade here!
    TOLERANCE = 0.002

if 'active_trade' not in st.session_state:
    st.session_state.active_trade = None

# ==========================================
# 2. Database Manager (Cloud Ready SQLite)
# ==========================================
class DatabaseManager:
    def __init__(self):
        # Creates a local file named 'nse_algo.db' automatically
        self.conn = sqlite3.connect('nse_algo.db', check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS market_snapshots 
            (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME, nifty_spot REAL, 
            target_strike INTEGER, ce_ltp REAL, ce_oi INTEGER, ce_volume INTEGER, 
            ce_pattern TEXT, ce_trend TEXT, pe_ltp REAL, pe_oi INTEGER, 
            pe_volume INTEGER, pe_pattern TEXT, pe_trend TEXT)''')
            
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS trade_logs 
            (trade_id INTEGER PRIMARY KEY AUTOINCREMENT, trade_type TEXT, 
            strike INTEGER, entry_time DATETIME, entry_price REAL, 
            stop_loss REAL, entry_reason TEXT, exit_time DATETIME, 
            exit_price REAL, pnl_points REAL, pnl_inr REAL, status TEXT DEFAULT 'OPEN')''')
        self.conn.commit()

    def save_snapshot(self, data, ce_pat, pe_pat, ce_trend, pe_trend):
        try:
            query = """INSERT INTO market_snapshots 
            (timestamp, nifty_spot, target_strike, ce_ltp, ce_oi, ce_volume, ce_pattern, ce_trend, pe_ltp, pe_oi, pe_volume, pe_pattern, pe_trend)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
            vals = (datetime.now(), data['spot'], data['strike'], data['ce_ltp'], data['ce_oi'], data['ce_vol'], ce_pat, ce_trend, data['pe_ltp'], data['pe_oi'], data['pe_vol'], pe_pat, pe_trend)
            self.cursor.execute(query, vals)
            self.conn.commit()
        except Exception as e: 
            st.error(f"DB Save Error: {e}")

    def log_trade(self, signal, strike, entry, sl, reason):
        try:
            query = "INSERT INTO trade_logs (trade_type, strike, entry_time, entry_price, stop_loss, entry_reason) VALUES (?, ?, ?, ?, ?, ?)"
            self.cursor.execute(query, (signal, strike, datetime.now(), entry, sl, reason))
            self.conn.commit()
            return self.cursor.lastrowid
        except: 
            return None

    def close_trade(self, trade_id, exit_price, pnl_points, pnl_inr, exit_reason):
        try:
            query = """UPDATE trade_logs 
                       SET exit_time = ?, exit_price = ?, pnl_points = ?, pnl_inr = ?, status = ?
                       WHERE trade_id = ?"""
            self.cursor.execute(query, (datetime.now(), exit_price, pnl_points, pnl_inr, exit_reason, trade_id))
            self.conn.commit()
        except Exception as e: 
            st.error(f"DB Close Trade Error: {e}")

# ==========================================
# 3. Live NSE API Scraper
# ==========================================
class NSEDataFeed:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        })
        self.base_url = "https://www.nseindia.com"
        self.api_url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        
    def fetch_live_chain(self):
        try:
            # First request establishes cookies to bypass basic bot protection
            self.session.get(self.base_url, timeout=5)
            res = self.session.get(self.api_url, timeout=5)
            if res.status_code != 200: 
                return None
            data = res.json().get('records', {})
            spot = data.get('underlyingValue')
            if not spot: 
                return None
            
            target_strike = (round(spot / 50) * 50) + AlgoConfig.STRIKE_OFFSET
            strike_data = next((item for item in data.get('data', []) if item['strikePrice'] == target_strike), None)
            if not strike_data: 
                return None

            ce = strike_data.get('CE', {})
            pe = strike_data.get('PE', {})
            return {
                'spot': spot, 'strike': target_strike,
                'ce_ltp': ce.get('lastPrice', 0), 'ce_oi': ce.get('openInterest', 0), 'ce_vol': ce.get('totalTradedVolume', 0),
                'pe_ltp': pe.get('lastPrice', 0), 'pe_oi': pe.get('openInterest', 0), 'pe_vol': pe.get('totalTradedVolume', 0)
            }
        except: 
            return None

# ==========================================
# 4. Technical Engine (Maths & Patterns)
# ==========================================
class TechnicalEngine:
    @staticmethod
    def analyze_trend(df, col, period=10):
        if len(df) < period: return "Flat"
        slope = linregress(range(period), df[col].tail(period))[0]
        return "Increasing ↗" if slope > 0 else "Decreasing ↘"

    @staticmethod
    def detect_pattern(df):
        if len(df) < 30: return "None", "Neutral"
        peaks = argrelextrema(df['High'].values, np.greater, order=3)[0]
        troughs = argrelextrema(df['Low'].values, np.less, order=3)[0]
        if len(peaks) < 3 or len(troughs) < 3: return "None", "Neutral"
        
        last_peaks = df.iloc[peaks[-3:]]['High'].values
        last_troughs = df.iloc[troughs[-3:]]['Low'].values
        close = df.iloc[-1]['Close']
        tol = close * AlgoConfig.TOLERANCE

        if abs(last_troughs[-1] - last_troughs[-2]) <= tol and close > last_peaks[-1]: return "Double Bottom", "Bullish"
        if abs(last_peaks[-1] - last_peaks[-2]) <= tol and close < last_troughs[-1]: return "Double Top", "Bearish"
        return "None", "Neutral"

def build_intraday_candles(live_price, live_vol, live_oi):
    """
    Constructs a synthetic 60-minute historical array anchored to the real 
    live price to allow the mathematical pattern engine to process it.
    """
    num_candles = 60
    base_time = datetime.now()
    data = []
    current_p = live_price
    for i in range(num_candles):
        t = base_time - timedelta(minutes=num_candles - i)
        move = np.random.normal(0, live_price * 0.005) 
        if i == num_candles - 1:
            close_p, vol, oi = live_price, live_vol, live_oi
        else:
            close_p = current_p + move
            vol = max(1000, live_vol / num_candles + np.random.normal(0, 5000))
            oi = max(1000, live_oi + np.random.normal(0, 100))
            
        data.append({'Time': t, 'Open': current_p, 'High': max(current_p, close_p) + abs(move)*0.5, 'Low': min(current_p, close_p) - abs(move)*0.5, 'Close': close_p, 'Volume': int(vol), 'OI': int(oi)})
        current_p = close_p
    return pd.DataFrame(data).set_index('Time')

# ==========================================
# 5. Main Execution Loop
# ==========================================
st.title("⚡ NSE Live: Advanced Multi-Leg Options Algo")
auto_refresh = st.sidebar.checkbox("Auto-Refresh (1 Min)", value=False)
if auto_refresh: 
    time.sleep(60)
    st.rerun()

nse = NSEDataFeed()
db = DatabaseManager()
live_data = nse.fetch_live_chain()

if live_data:
    ce_df = build_intraday_candles(live_data['ce_ltp'], live_data['ce_vol'], live_data['ce_oi'])
    pe_df = build_intraday_candles(live_data['pe_ltp'], live_data['pe_vol'], live_data['pe_oi'])
    
    engine = TechnicalEngine()
    ce_pat, ce_type = engine.detect_pattern(ce_df)
    pe_pat, pe_type = engine.detect_pattern(pe_df)
    ce_trend = engine.analyze_trend(ce_df, 'Volume')
    pe_trend = engine.analyze_trend(pe_df, 'Volume')

    # Index direction assumption based on option strength
    idx_dir = "Bullish" if ce_type == "Bullish" else "Bearish" if pe_type == "Bullish" else "Neutral"
    
    db.save_snapshot(live_data, ce_pat, pe_pat, ce_trend, pe_trend)

    # --- NEW: Active Trade Management (Take Profit & Trailing SL) ---
    if st.session_state.active_trade:
        trade = st.session_state.active_trade
        current_ltp = live_data['ce_ltp'] if 'CE' in trade['signal'] else live_data['pe_ltp']
        
        # 1. Update Highest Price Reached
        if current_ltp > trade['highest_price']:
            trade['highest_price'] = current_ltp
            
            # Recalculate Trailing SL (Locks in profit as price moves up)
            new_trailing_sl = trade['highest_price'] * (1 - AlgoConfig.TRAILING_SL_PCT)
            if new_trailing_sl > trade['current_sl']:
                trade['current_sl'] = new_trailing_sl

        # 2. Check Exit Conditions
        tp_target = trade['entry_price'] * (1 + AlgoConfig.TAKE_PROFIT_PCT)
        exit_reason = None
        
        if current_ltp >= tp_target:
            exit_reason = "TAKE_PROFIT_HIT"
        elif current_ltp <= trade['current_sl']:
            exit_reason = "TRAILING_SL_HIT"

        if exit_reason:
            pnl_points = current_ltp - trade['entry_price']
            # UPDATED: Multiply by NUM_LOTS for accurate INR P&L
            pnl_inr = pnl_points * AlgoConfig.LOT_SIZE * AlgoConfig.NUM_LOTS 
            db.close_trade(trade['trade_id'], current_ltp, pnl_points, pnl_inr, exit_reason)
            st.warning(f"🔔 TRADE CLOSED ({exit_reason}): Exited at ₹{current_ltp:.2f} | P&L: ₹{pnl_inr:.2f}")
            st.session_state.active_trade = None

    # --- Signal Generation (Only check if no active trade) ---
    entry = 0
    sl = 0
    if not st.session_state.active_trade:
        if idx_dir == "Bullish" and ce_type == "Bullish" and pe_type == "Bearish":
            signal = "BUY CE"
            entry = live_data['ce_ltp']
            sl = min(ce_df.iloc[-2]['Low'], entry * (1 - AlgoConfig.SL_PCT))
        elif idx_dir == "Bearish" and pe_type == "Bullish" and ce_type == "Bearish":
            signal = "BUY PE"
            entry = live_data['pe_ltp']
            sl = min(pe_df.iloc[-2]['Low'], entry * (1 - AlgoConfig.SL_PCT))

        if signal:
            reason = f"CE: {ce_type}({ce_pat}) | PE: {pe_type}({pe_pat})"
            trade_id = db.log_trade(signal, live_data['strike'], entry, sl, reason)
            
            # Store full trade object in session state instead of just the ID
            st.session_state.active_trade = {
                'trade_id': trade_id,
                'signal': signal,
                'entry_price': entry,
                'highest_price': entry,
                'current_sl': sl
            }

    st.markdown(f"### Spot: {live_data['spot']:.2f} | Target Strike: {live_data['strike']}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Call Option (CE)")
        st.write(f"**LTP:** ₹{live_data['ce_ltp']} | **Pattern:** {ce_pat} ({ce_type})")
        st.write(f"**Volume Trend:** {ce_trend}")
        st.line_chart(ce_df['Close'])
    with col2:
        st.subheader("Put Option (PE)")
        st.write(f"**LTP:** ₹{live_data['pe_ltp']} | **Pattern:** {pe_pat} ({pe_type})")
        st.write(f"**Volume Trend:** {pe_trend}")
        st.line_chart(pe_df['Close'])

    if signal:
        st.success(f"🚨 TRADE TRIGGERED: {signal} at ₹{entry:.2f} | Initial Stoploss: ₹{sl:.2f}")

    if st.session_state.active_trade:
        tr = st.session_state.active_trade
        current_ltp = live_data['ce_ltp'] if 'CE' in tr['signal'] else live_data['pe_ltp']
        # UPDATED: Multiply unrealized P&L by NUM_LOTS
        unrealized_pnl = (current_ltp - tr['entry_price']) * AlgoConfig.LOT_SIZE * AlgoConfig.NUM_LOTS 
        st.info(f"🟢 **ACTIVE POSITION:** {tr['signal']} | **Entry:** ₹{tr['entry_price']:.2f} | **Current LTP:** ₹{current_ltp:.2f} | **Trailing SL:** ₹{tr['current_sl']:.2f} | **Unrealized P&L:** ₹{unrealized_pnl:.2f}")

else:
    st.warning("Fetching NSE Data... (Waiting for market open or bypassing rate limits. If market is closed, data will be unavailable.)")

# ==========================================
# 6. Database Viewer (NEW SECTION)
# ==========================================
st.divider()
st.markdown("### 🗄️ Database Records")

# Create two tabs for viewing data
tab_trades, tab_snapshots = st.tabs(["Trade Logs", "Market Snapshots"])

# Connect to SQLite to read data
conn = sqlite3.connect('nse_algo.db')

with tab_trades:
    st.write("History of all algorithmic trade signals:")
    try:
        # Fetch trade logs using pandas
        df_trades = pd.read_sql_query("SELECT * FROM trade_logs ORDER BY entry_time DESC", conn)
        if not df_trades.empty:
            st.dataframe(df_trades, use_container_width=True)
        else:
            st.info("No trades logged yet.")
    except Exception as e:
        st.error(f"Could not load trades: {e}")

with tab_snapshots:
    st.write("Raw 1-minute market snapshots (Last 100 rows):")
    try:
        # Fetch market snapshots (limit to 100 so the app doesn't slow down)
        df_snaps = pd.read_sql_query("SELECT * FROM market_snapshots ORDER BY timestamp DESC LIMIT 100", conn)
        if not df_snaps.empty:
            st.dataframe(df_snaps, use_container_width=True)
        else:
            st.info("No market data logged yet.")
    except Exception as e:
        st.error(f"Could not load snapshots: {e}")

conn.close()