import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from scipy.stats import linregress, norm
from datetime import datetime, timedelta
import pytz
import sqlite3
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

# ==========================================
# 1. Configuration & State Initialization
# ==========================================
st.set_page_config(page_title="Nifty Master Algo: Ultra-Aggressive", layout="wide", page_icon="⚡")
IST = pytz.timezone('Asia/Kolkata')

# --- FAILSAFE INITIALIZATION ---
if 'active_trade' not in st.session_state:
    st.session_state.active_trade = None

class AlgoConfig:
    STRIKE_OFFSET = 100     # Default offset from ATM
    SL_PCT = 0.20           
    TAKE_PROFIT_PCT = 0.40      
    TRAILING_SL_PCT = 0.05  
    TRAILING_ACTIVATION_PCT = 0.10 
    LOT_SIZE = 65           # PERMANENT
    NUM_LOTS = 3            # PERMANENT                
    TOLERANCE = 0.002       
    
    # --- Logic Settings ---
    BREAKOUT_WINDOW = 10        
    VOL_MA_PERIOD = 20          
    
    # --- Pricing Parameter Defaults ---
    RISK_FREE_RATE = 0.07       
    DEFAULT_CE_IV = 0.23        
    DEFAULT_PE_IV = 0.29        
    EXPIRY_DATE = "2026-04-07"  
    DEFAULT_FUTURES_PREMIUM = 100.0      

# ==========================================
# 2. Database Manager
# ==========================================
class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect('nse_algo.db', check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_tables()
        self.migrate_schema() 

    def create_tables(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS market_snapshots 
            (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME, nifty_spot REAL, 
            target_strike INTEGER, ce_ltp REAL, ce_oi INTEGER, ce_volume INTEGER, 
            ce_pattern TEXT, ce_trend TEXT, pe_ltp REAL, pe_oi INTEGER, 
            pe_volume INTEGER, pe_pattern TEXT, pe_trend TEXT)''')
            
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS trade_logs 
            (trade_id INTEGER PRIMARY KEY AUTOINCREMENT, trade_type TEXT, 
            strike INTEGER, entry_time DATETIME, entry_price REAL, 
            stop_loss REAL, highest_price REAL, entry_reason TEXT, exit_time DATETIME, 
            exit_price REAL, pnl_points REAL, pnl_inr REAL, 
            status TEXT DEFAULT 'OPEN')''')
        self.conn.commit()

    def migrate_schema(self):
        try:
            self.cursor.execute("ALTER TABLE trade_logs ADD COLUMN highest_price REAL")
            self.conn.commit()
        except: pass 

    def get_active_trade_from_db(self):
        try:
            query = "SELECT * FROM trade_logs WHERE status = 'OPEN' LIMIT 1"
            df = pd.read_sql_query(query, self.conn)
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
        except: return None

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
            query = "INSERT INTO trade_logs (trade_type, strike, entry_time, entry_price, stop_loss, highest_price, entry_reason, status) VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN')"
            self.cursor.execute(query, (signal, strike, datetime.now(), entry, sl, entry, reason))
            self.conn.commit()
            return self.cursor.lastrowid
        except: return None

    def update_trade_sl(self, trade_id, highest_price, stop_loss):
        try:
            query = "UPDATE trade_logs SET highest_price = ?, stop_loss = ? WHERE trade_id = ?"
            self.cursor.execute(query, (highest_price, stop_loss, trade_id))
            self.conn.commit()
        except: pass

    def close_trade(self, trade_id, exit_price, pnl_points, pnl_inr, exit_reason):
        try:
            query = """UPDATE trade_logs 
                       SET exit_time = ?, exit_price = ?, pnl_points = ?, pnl_inr = ?, status = ?
                       WHERE trade_id = ?"""
            self.cursor.execute(query, (datetime.now(), exit_price, pnl_points, pnl_inr, exit_reason, trade_id))
            self.conn.commit()
        except: pass

# ==========================================
# 3. Quantitative Engine: Black-76 Pricing
# ==========================================
class BlackScholes:
    @staticmethod
    def price(F, K, T, r, sigma, option_type='call'):
        T = max(T, 1e-5) 
        d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

# ==========================================
# 4. Data Layer (YFinance + Synthetic Pipeline)
# ==========================================
@st.cache_data(ttl=30)
def get_nifty_spot_data():
    try:
        nifty = yf.Ticker("^NSEI")
        df = nifty.history(period="10d", interval="5m")
        return df
    except: return pd.DataFrame()

class SyntheticDataFeed:
    def fetch_live_options(self, spot_price, futures_premium, ce_iv, pe_iv, strike_offset):
        expiry_date = datetime.strptime(AlgoConfig.EXPIRY_DATE, "%Y-%m-%d").date()
        current_date = datetime.now(IST).date()
        dte = max((expiry_date - current_date).days, 1) 
        T = dte / 365.0
        r = AlgoConfig.RISK_FREE_RATE
        
        futures_price = spot_price + futures_premium
        # Calculate Target Strike using dynamic offset
        target_strike = (int(round(futures_price / 50.0)) * 50) + strike_offset
        
        ce_theo = BlackScholes.price(futures_price, target_strike, T, r, ce_iv, 'call')
        pe_theo = BlackScholes.price(futures_price, target_strike, T, r, pe_iv, 'put')
        
        return {
            'spot': spot_price, 'futures_price': futures_price, 'strike': target_strike, 'dte': dte,
            'ce_ltp': round(max(0.5, ce_theo + np.random.normal(0, 1.5)), 2), 
            'ce_oi': int(2500000 + np.random.normal(0, 15000)), 'ce_vol': int(1500000 + np.random.normal(0, 25000)),
            'pe_ltp': round(max(0.5, pe_theo + np.random.normal(0, 1.5)), 2), 
            'pe_oi': int(2000000 + np.random.normal(0, 15000)), 'pe_vol': int(1200000 + np.random.normal(0, 25000)),
            'feed_mode': '🟢 BLACK-76 OPTIONS FEED'
        }

# ==========================================
# 5. Technical Engine (Patterns & Trends)
# ==========================================
class TechnicalEngine:
    @staticmethod
    def check_breakout(df, window=10):
        if len(df) < window + 1: return "Neutral"
        lookback = df.iloc[-(window+1):-1]
        resistance = lookback['High'].max()
        support = lookback['Low'].min()
        current_close = df.iloc[-1]['Close']
        if current_close > resistance: return "Bullish Breakout"
        if current_close < support: return "Bearish Breakdown"
        return "Neutral"

    @staticmethod
    def detect_pattern(df, order=1): 
        if len(df) < 20: return "None", "Neutral"
        peaks = argrelextrema(df['High'].values, np.greater, order=order)[0]
        troughs = argrelextrema(df['Low'].values, np.less, order=order)[0]
        if len(peaks) < 2 or len(troughs) < 2: return "None", "Neutral"
        
        last_peaks = df.iloc[peaks[-3:]]['High'].values
        last_troughs = df.iloc[troughs[-3:]]['Low'].values
        close = df.iloc[-1]['Close']
        tol = close * AlgoConfig.TOLERANCE

        if abs(last_troughs[-1] - last_troughs[-2]) <= tol and close > last_troughs[-1]:
            return "Double Bottom (Fast)", "Bullish"
        if abs(last_peaks[-1] - last_peaks[-2]) <= tol and close < last_peaks[-1]:
            return "Double Top (Fast)", "Bearish"
        return "None", "Neutral"

def build_intraday_candles(live_price, live_vol, live_oi):
    num_candles = 60
    base_time = datetime.now()
    data = []
    current_p = live_price
    for i in range(num_candles):
        t = base_time - timedelta(minutes=num_candles - i)
        move = np.random.normal(0, live_price * 0.005) 
        close_p = current_p + move if i != num_candles - 1 else live_price
        data.append({'Time': t, 'Open': current_p, 'High': max(current_p, close_p) + abs(move)*0.5, 'Low': min(current_p, close_p) - abs(move)*0.5, 'Close': close_p, 'Volume': int(live_vol/num_candles), 'OI': int(live_oi)})
        current_p = close_p
    return pd.DataFrame(data).set_index('Time').sort_index()

# ==========================================
# 6. Main Execution Loop
# ==========================================
st.sidebar.title("🛠️ Settings & Controls")

st.sidebar.markdown("### 📊 Pricing Inputs")
live_ce_iv = st.sidebar.slider("Call IV (CE %)", 5, 100, int(AlgoConfig.DEFAULT_CE_IV * 100), step=1) / 100.0
live_pe_iv = st.sidebar.slider("Put IV (PE %)", 5, 100, int(AlgoConfig.DEFAULT_PE_IV * 100), step=1) / 100.0
live_fut_prem = st.sidebar.slider("Futures Premium (Pts)", 0, 500, int(AlgoConfig.DEFAULT_FUTURES_PREMIUM), step=5)

# --- NEW: STRIKE PRICE CONTROL ---
st.sidebar.markdown("### 🎯 Selection")
live_strike_offset = st.sidebar.slider("Strike Offset (from ATM)", -500, 500, int(AlgoConfig.STRIKE_OFFSET), step=50, help="Shift target strike. Positive = OTM Call / ITM Put.")

st.sidebar.divider()
auto_refresh = st.sidebar.checkbox("Auto-Refresh (1 Min)", value=True)

col_title, col_btn = st.columns([5, 1])
with col_title:
    st.title("⚡ Master Algo: ULTRA-AGGRESSIVE")
with col_btn:
    st.write(""); 
    if st.button("🔄 Refresh Data", use_container_width=True): st.rerun()

if auto_refresh:
    st_autorefresh(interval=60000, limit=None, key="algo_refresh")

db = DatabaseManager()
if st.session_state.active_trade is None:
    st.session_state.active_trade = db.get_active_trade_from_db()

nifty_df = get_nifty_spot_data()

if not nifty_df.empty:
    current_spot = nifty_df['Close'].iloc[-1]
    
    # Indicators calculation
    nifty_df['EMA_20'] = nifty_df['Close'].ewm(span=20, adjust=False).mean()
    nifty_df['EMA_50'] = nifty_df['Close'].ewm(span=50, adjust=False).mean()
    nifty_df['EMA_200'] = nifty_df['Close'].ewm(span=200, adjust=False).mean()
    nifty_df['Vol_MA'] = nifty_df['Volume'].rolling(window=AlgoConfig.VOL_MA_PERIOD).mean()
    
    delta = nifty_df['Close'].diff()
    gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean(); avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    nifty_df['RSI'] = 100 - (100 / (1 + avg_gain / avg_loss))
    
    engine = TechnicalEngine()
    nifty_pat, nifty_pat_type = engine.detect_pattern(nifty_df, order=1)
    nifty_breakout = engine.check_breakout(nifty_df, window=AlgoConfig.BREAKOUT_WINDOW)
    
    # --- Index Bias Logic ---
    latest = nifty_df.iloc[-1]
    prev = nifty_df.iloc[-2]
    trend_score = 0
    if latest['EMA_20'] > latest['EMA_50'] > latest['EMA_200']: trend_score += 1
    elif latest['EMA_20'] < latest['EMA_50'] < latest['EMA_200']: trend_score -= 1
    if latest['RSI'] > 50: trend_score += 1 
    elif latest['RSI'] < 50: trend_score -= 1
    vol_confirmed = latest['Volume'] > latest['Vol_MA']
    if prev['Close'] < prev['EMA_20'] and latest['Close'] > latest['EMA_20'] and vol_confirmed: trend_score += 1 
    elif prev['Close'] > prev['EMA_20'] and latest['Close'] < latest['EMA_20'] and vol_confirmed: trend_score -= 1 
    
    idx_dir = "Bullish" if ("Bullish" in nifty_breakout or nifty_pat_type == "Bullish" or trend_score >= 1) else "Bearish" if ("Bearish" in nifty_breakout or nifty_pat_type == "Bearish" or trend_score <= -1) else "Neutral"

    # Pass slider values including STRIKE OFFSET into the data feed
    nse = SyntheticDataFeed()
    live_data = nse.fetch_live_options(current_spot, live_fut_prem, live_ce_iv, live_pe_iv, live_strike_offset)
    
    # --- ACTIVE TRADE MANAGEMENT ---
    floating_pnl = 0.0
    status_text = "WAITING"
    trade_just_closed = False 
    
    if st.session_state.active_trade:
        trade = st.session_state.active_trade
        current_ltp = live_data['ce_ltp'] if 'CE' in trade['trade_type'] else live_data['pe_ltp']
        entry_p = trade['entry_price']
        floating_pnl = (current_ltp - entry_p) * AlgoConfig.LOT_SIZE * AlgoConfig.NUM_LOTS
        status_text = f"ACTIVE: {trade['trade_type']}"

        highest_so_far = trade.get('highest_price') or entry_p
        is_trailing_active = current_ltp >= (entry_p * (1 + AlgoConfig.TRAILING_ACTIVATION_PCT))
        
        if is_trailing_active and current_ltp > highest_so_far:
            highest_so_far = current_ltp
            trade['highest_price'] = highest_so_far
            new_tsl = highest_so_far * (1 - AlgoConfig.TRAILING_SL_PCT)
            if new_tsl > trade['stop_loss']:
                trade['stop_loss'] = new_tsl
                db.update_trade_sl(trade['trade_id'], highest_so_far, trade['stop_loss'])
        
        tp_target = entry_p * (1 + AlgoConfig.TAKE_PROFIT_PCT)
        curr_sl = trade['stop_loss']
        exit_reason = "TAKE_PROFIT" if current_ltp >= tp_target else "STOP_LOSS" if current_ltp <= curr_sl else None
        
        if exit_reason:
            pts = current_ltp - entry_p
            db.close_trade(trade['trade_id'], current_ltp, pts, floating_pnl, exit_reason)
            st.session_state.active_trade = None
            trade_just_closed = True

    # --- TOP ROW: COMMAND CENTER METRICS ---
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Live Spot", f"₹{current_spot:.2f}")
    m2.metric("Index Pattern", nifty_pat, delta=nifty_pat_type)
    m3.metric("Trend Score", trend_score, delta=idx_dir)
    m4.metric("Active Strike", live_data['strike'])
    m5.metric("Status", status_text)
    m6.metric("Live P&L", f"₹{floating_pnl:.2f}")

    st.divider()

    # --- 2-DAY VIEW CHART ---
    nifty_dates = pd.to_datetime(nifty_df.index).date
    unique_dates = np.unique(nifty_dates)
    display_dates = unique_dates[-2:] if len(unique_dates) >= 2 else unique_dates
    nifty_display_df = nifty_df[np.isin(nifty_dates, display_dates)]

    st.markdown(f"### 📈 Nifty 50 Trend")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
    fig.add_trace(go.Candlestick(x=nifty_display_df.index, open=nifty_display_df['Open'], high=nifty_display_df['High'], low=nifty_display_df['Low'], close=nifty_display_df['Close'], name='Nifty'), row=1, col=1)
    fig.add_trace(go.Scatter(x=nifty_display_df.index, y=nifty_display_df['EMA_20'], line=dict(color='blue', width=1), name='EMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=nifty_display_df.index, y=nifty_display_df['EMA_200'], line=dict(color='red', width=1), name='EMA 200'), row=1, col=1)
    fig.add_trace(go.Bar(x=nifty_display_df.index, y=nifty_display_df['Volume'], name='Volume'), row=2, col=1)
    fig.add_trace(go.Scatter(x=nifty_display_df.index, y=nifty_display_df['RSI'], name='RSI'), row=3, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=[15.5, 9.25], pattern="hour"), dict(bounds=["sat", "mon"])])
    fig.update_layout(height=450, template='plotly_dark', xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    
    ce_df = build_intraday_candles(live_data['ce_ltp'], live_data['ce_vol'], live_data['ce_oi'])
    pe_df = build_intraday_candles(live_data['pe_ltp'], live_data['pe_vol'], live_data['pe_oi'])
    ce_pat, ce_type = engine.detect_pattern(ce_df, order=1)
    pe_pat, pe_type = engine.detect_pattern(pe_df, order=1)
    db.save_snapshot(live_data, ce_pat, pe_pat, "Flat", "Flat")

    # --- ENTRY GATE: ONLY 1 TRADE ALLOWED ---
    if st.session_state.active_trade is None and not trade_just_closed:
        signal = "BUY CE" if (idx_dir=="Bullish" and (ce_type=="Bullish" or pe_type=="Bearish")) else "BUY PE" if (idx_dir=="Bearish" and (pe_type=="Bullish" or ce_type=="Bearish")) else None
        if signal:
            entry = live_data['ce_ltp'] if "CE" in signal else live_data['pe_ltp']
            sl = entry * (1 - AlgoConfig.SL_PCT)
            reason = f"IDX:{nifty_pat}|CE:{ce_pat}|PE:{pe_pat}"
            t_id = db.log_trade(signal, live_data['strike'], entry, sl, reason)
            st.session_state.active_trade = {'trade_id':t_id, 'trade_type':signal, 'entry_price':entry, 'highest_price':entry, 'stop_loss':sl}

    c_ce, c_pe = st.columns(2)
    with c_ce:
        st.subheader(f"Call Option (CE) - Strike {live_data['strike']}")
        st.write(f"**LTP:** ₹{live_data['ce_ltp']} | **Pattern:** {ce_pat}")
        fig_ce = go.Figure(data=[go.Candlestick(x=ce_df.index, open=ce_df['Open'], high=ce_df['High'], low=ce_df['Low'], close=ce_df['Close'])])
        fig_ce.update_layout(height=280, template='plotly_dark', margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_ce, use_container_width=True)
    with c_pe:
        st.subheader(f"Put Option (PE) - Strike {live_data['strike']}")
        st.write(f"**LTP:** ₹{live_data['pe_ltp']} | **Pattern:** {pe_pat}")
        fig_pe = go.Figure(data=[go.Candlestick(x=pe_df.index, open=pe_df['Open'], high=pe_df['High'], low=pe_df['Low'], close=pe_df['Close'])])
        fig_pe.update_layout(height=280, template='plotly_dark', margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_pe, use_container_width=True)

    if st.session_state.active_trade:
        tr = st.session_state.active_trade
        curr = live_data['ce_ltp'] if 'CE' in tr['trade_type'] else live_data['pe_ltp']
        col_info, col_manual = st.columns([4, 1])
        col_info.info(f"🟢 **ACTIVE:** {tr['trade_type']} | **Entry:** ₹{tr['entry_price']:.2f} | **LTP:** ₹{curr:.2f} | **SL:** ₹{tr['stop_loss']:.2f} | **P&L:** ₹{(curr-tr['entry_price'])*AlgoConfig.LOT_SIZE*AlgoConfig.NUM_LOTS:.2f}")
        if col_manual.button("❌ Square Off", use_container_width=True):
            pts = curr - tr['entry_price']
            final_pnl = pts * AlgoConfig.LOT_SIZE * AlgoConfig.NUM_LOTS
            db.close_trade(tr['trade_id'], curr, pts, final_pnl, "MANUAL_EXIT")
            st.session_state.active_trade = None
            st.rerun()

st.divider()
st.markdown("### 🗄️ Database Records")
t1, t2 = st.tabs(["Trade Logs", "Market Snapshots"])
conn = sqlite3.connect('nse_algo.db')
with t1:
    try: st.dataframe(pd.read_sql_query("SELECT * FROM trade_logs ORDER BY entry_time DESC", conn), use_container_width=True)
    except: st.info("No trades logged yet.")
with t2:
    try: st.dataframe(pd.read_sql_query("SELECT * FROM market_snapshots ORDER BY timestamp DESC LIMIT 100", conn), use_container_width=True)
    except: st.info("No snapshots logged yet.")
conn.close()