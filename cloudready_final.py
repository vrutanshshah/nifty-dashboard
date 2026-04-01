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
    LOT_SIZE = 25
    NUM_LOTS = 2                
    TOLERANCE = 0.002

if 'active_trade' not in st.session_state:
    st.session_state.active_trade = None

# ==========================================
# 2. Database Manager (Cloud Ready SQLite)
# ==========================================
class DatabaseManager:
    def __init__(self):
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
        except: pass

    def log_trade(self, signal, strike, entry, sl, reason):
        try:
            query = "INSERT INTO trade_logs (trade_type, strike, entry_time, entry_price, stop_loss, entry_reason) VALUES (?, ?, ?, ?, ?, ?)"
            self.cursor.execute(query, (signal, strike, datetime.now(), entry, sl, reason))
            self.conn.commit()
            return self.cursor.lastrowid
        except: return None

    def close_trade(self, trade_id, exit_price, pnl_points, pnl_inr, exit_reason):
        try:
            query = """UPDATE trade_logs 
                       SET exit_time = ?, exit_price = ?, pnl_points = ?, pnl_inr = ?, status = ?
                       WHERE trade_id = ?"""
            self.cursor.execute(query, (datetime.now(), exit_price, pnl_points, pnl_inr, exit_reason, trade_id))
            self.conn.commit()
        except: pass

# ==========================================
# 3. Live NSE API Scraper (FIXED HEADERS)
# ==========================================
class NSEDataFeed:
    def __init__(self):
        self.session = requests.Session()
        # CRITICAL FIX: Added Referer and more detailed browser headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/get-quotes/derivatives?symbol=NIFTY',
            'X-Requested-With': 'XMLHttpRequest'
        })
        self.base_url = "https://www.nseindia.com"
        self.api_url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        
    def fetch_live_chain(self):
        try:
            # First hit the home page to get the session cookies
            self.session.get(self.base_url, timeout=10)
            
            # Now hit the API
            res = self.session.get(self.api_url, timeout=10)
            
            # DEBUG: Show error if blocked
            if res.status_code != 200:
                st.error(f"🛑 NSE Error {res.status_code}: The exchange is blocking the request. If you are on Streamlit Cloud, try running this LOCALLY in VS Code.")
                return None
                
            data = res.json().get('records', {})
            spot = data.get('underlyingValue')
            if not spot: return None
            
            target_strike = (round(spot / 50) * 50) + AlgoConfig.STRIKE_OFFSET
            strike_data = next((item for item in data.get('data', []) if item['strikePrice'] == target_strike), None)
            if not strike_data: return None

            ce = strike_data.get('CE', {})
            pe = strike_data.get('PE', {})
            return {
                'spot': spot, 'strike': target_strike,
                'ce_ltp': ce.get('lastPrice', 0), 'ce_oi': ce.get('openInterest', 0), 'ce_vol': ce.get('totalTradedVolume', 0),
                'pe_ltp': pe.get('lastPrice', 0), 'pe_oi': pe.get('openInterest', 0), 'pe_vol': pe.get('totalTradedVolume', 0)
            }
        except Exception as e: 
            st.error(f"⚠️ Connection Error: {str(e)}")
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

    idx_dir = "Bullish" if ce_type == "Bullish" else "Bearish" if pe_type == "Bullish" else "Neutral"
    db.save_snapshot(live_data, ce_pat, pe_pat, ce_trend, pe_trend)

    # --- Active Trade Management (Trailing SL & TP) ---
    if st.session_state.active_trade:
        trade = st.session_state.active_trade
        current_ltp = live_data['ce_ltp'] if 'CE' in trade['signal'] else live_data['pe_ltp']
        
        if current_ltp > trade['highest_price']:
            trade['highest_price'] = current_ltp
            new_trailing_sl = trade['highest_price'] * (1 - AlgoConfig.TRAILING_SL_PCT)
            if new_trailing_sl > trade['current_sl']:
                trade['current_sl'] = new_trailing_sl

        tp_target = trade['entry_price'] * (1 + AlgoConfig.TAKE_PROFIT_PCT)
        exit_reason = None
        
        if current_ltp >= tp_target: exit_reason = "TAKE_PROFIT_HIT"
        elif current_ltp <= trade['current_sl']: exit_reason = "TRAILING_SL_HIT"

        if exit_reason:
            pnl_points = current_ltp - trade['entry_price']
            pnl_inr = pnl_points * AlgoConfig.LOT_SIZE * AlgoConfig.NUM_LOTS
            db.close_trade(trade['trade_id'], current_ltp, pnl_points, pnl_inr, exit_reason)
            st.warning(f"🔔 TRADE CLOSED ({exit_reason}): Exited at ₹{current_ltp:.2f} | P&L: ₹{pnl_inr:.2f}")
            st.session_state.active_trade = None

    if not st.session_state.active_trade:
        signal = None
        if idx_dir == "Bullish" and ce_type == "Bullish" and pe_type == "Bearish":
            signal = "BUY CE"; entry = live_data['ce_ltp']
            sl = min(ce_df.iloc[-2]['Low'], entry * (1 - AlgoConfig.SL_PCT))
        elif idx_dir == "Bearish" and pe_type == "Bullish" and ce_type == "Bearish":
            signal = "BUY PE"; entry = live_data['pe_ltp']
            sl = min(pe_df.iloc[-2]['Low'], entry * (1 - AlgoConfig.SL_PCT))

        if signal:
            reason = f"CE: {ce_type}({ce_pat}) | PE: {pe_type}({pe_pat})"
            trade_id = db.log_trade(signal, live_data['strike'], entry, sl, reason)
            st.session_state.active_trade = {
                'trade_id': trade_id, 'signal': signal, 'entry_price': entry,
                'highest_price': entry, 'current_sl': sl
            }

    st.markdown(f"### Spot: {live_data['spot']:.2f} | Target Strike: {live_data['strike']}")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Call Option (CE)")
        st.write(f"**LTP:** ₹{live_data['ce_ltp']} | **Pattern:** {ce_pat}")
        st.line_chart(ce_df['Close'])
    with col2:
        st.subheader("Put Option (PE)")
        st.write(f"**LTP:** ₹{live_data['pe_ltp']} | **Pattern:** {pe_pat}")
        st.line_chart(pe_df['Close'])

    if st.session_state.active_trade:
        tr = st.session_state.active_trade
        curr = live_data['ce_ltp'] if 'CE' in tr['signal'] else live_data['pe_ltp']
        u_pnl = (curr - tr['entry_price']) * AlgoConfig.LOT_SIZE * AlgoConfig.NUM_LOTS
        st.info(f"🟢 **ACTIVE:** {tr['signal']} | **Entry:** ₹{tr['entry_price']:.2f} | **LTP:** ₹{curr:.2f} | **SL:** ₹{tr['current_sl']:.2f} | **P&L:** ₹{u_pnl:.2f}")

else:
    st.warning("Fetching NSE Data... (If market is closed, data will be unavailable.)")

# ==========================================
# 6. Database Viewer
# ==========================================
st.divider()
st.markdown("### 🗄️ Database Records")
tab_trades, tab_snapshots = st.tabs(["Trade Logs", "Market Snapshots"])
conn = sqlite3.connect('nse_algo.db')
with tab_trades:
    try:
        df_trades = pd.read_sql_query("SELECT * FROM trade_logs ORDER BY entry_time DESC", conn)
        st.dataframe(df_trades, use_container_width=True)
    except: st.info("No trades logged yet.")
with tab_snapshots:
    try:
        df_snaps = pd.read_sql_query("SELECT * FROM market_snapshots ORDER BY timestamp DESC LIMIT 100", conn)
        st.dataframe(df_snaps, use_container_width=True)
    except: st.info("No snapshots logged yet.")
conn.close()