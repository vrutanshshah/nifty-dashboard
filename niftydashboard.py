import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from datetime import datetime
import pytz
import time

# ==========================================
# 1. Page Configuration & State
# ==========================================
st.set_page_config(page_title="Nifty Options Algo Terminal", layout="wide", page_icon="📈")

# Initialize Session State for Paper Trading
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'active_trade' not in st.session_state:
    st.session_state.active_trade = None
if 'total_pnl' not in st.session_state:
    st.session_state.total_pnl = 0.0

IST = pytz.timezone('Asia/Kolkata')

# ==========================================
# 2. Technical Analysis Logic
# ==========================================
def calculate_indicators(df):
    if len(df) < 100: return df
        
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def detect_patterns(df):
    if len(df) < 50: return {"pattern": "None", "signal": 0}
    peaks = argrelextrema(df['High'].values, np.greater, order=5)[0]
    troughs = argrelextrema(df['Low'].values, np.less, order=5)[0]
    
    if len(peaks) < 3 or len(troughs) < 3: return {"pattern": "None", "signal": 0}
    
    last_peaks = df.iloc[peaks[-3:]]['High'].values
    last_troughs = df.iloc[troughs[-3:]]['Low'].values
    current_close = df.iloc[-1]['Close']
    tol = current_close * 0.0015 # 0.15% tolerance

    if abs(last_peaks[-1] - last_peaks[-2]) <= tol and current_close < last_troughs[-1]:
        return {"pattern": "Double Top", "signal": -1}
    if abs(last_troughs[-1] - last_troughs[-2]) <= tol and current_close > last_peaks[-1]:
        return {"pattern": "Double Bottom", "signal": 1}
    return {"pattern": "None", "signal": 0}

# ==========================================
# 3. Data Fetching
# ==========================================
@st.cache_data(ttl=60) # Cache data for 60 seconds to prevent API spam
def fetch_live_data():
    ticker = yf.Ticker("^NSEI")
    df = ticker.history(period="5d", interval="5m")
    if not df.empty:
        df = calculate_indicators(df)
    return df

# ==========================================
# 4. UI Layout & Rendering
# ==========================================
st.title("🚀 Nifty 50 Options Algorithmic Dashboard")
st.markdown("Live 5-Minute Timeframe | EMA Confluence + RSI + Pattern Recognition")

# Auto-refresh logic
col_auto, col_status = st.columns([1, 4])
with col_auto:
    auto_refresh = st.checkbox("Auto-Refresh (Live Market)", value=True)

df = fetch_live_data()

if df.empty:
    st.error("Failed to fetch data from Yahoo Finance. Check your internet connection.")
    st.stop()

latest = df.iloc[-1]
current_spot = latest['Close']
timestamp = df.index[-1].astimezone(IST).strftime('%I:%M %p')

# --- Algo Scoring Logic ---
score = 0
reasons = []

if latest['EMA_20'] > latest['EMA_50'] > latest['EMA_100']: 
    score += 1; reasons.append("Bullish EMAs")
elif latest['EMA_20'] < latest['EMA_50'] < latest['EMA_100']: 
    score -= 1; reasons.append("Bearish EMAs")
    
if 55 < latest['RSI'] < 70: 
    score += 1; reasons.append("Bullish RSI")
elif 30 < latest['RSI'] < 45: 
    score -= 1; reasons.append("Bearish RSI")
    
pat_data = detect_patterns(df)
if pat_data['signal'] != 0:
    score += (pat_data['signal'] * 2)
    reasons.append(f"Pattern: {pat_data['pattern']}")

# --- Paper Trading Logic ---
if st.session_state.active_trade:
    trade = st.session_state.active_trade
    is_reversed = (trade['type'] == 'CE' and score < 0) or (trade['type'] == 'PE' and score > 0)
    
    if is_reversed:
        points = (current_spot - trade['entry']) if trade['type'] == 'CE' else (trade['entry'] - current_spot)
        pnl = points * 0.5 * 25 # Est. Delta 0.5, Lot Size 25
        st.session_state.total_pnl += pnl
        st.session_state.trade_history.insert(0, {
            'time': timestamp, 'type': trade['type'], 'entry': trade['entry'], 
            'exit': current_spot, 'pnl': pnl
        })
        st.session_state.active_trade = None

if not st.session_state.active_trade:
    if score >= 2:
        st.session_state.active_trade = {'type': 'CE', 'entry': current_spot, 'time': timestamp}
    elif score <= -2:
        st.session_state.active_trade = {'type': 'PE', 'entry': current_spot, 'time': timestamp}

# --- Display Metrics ---
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Nifty Spot", f"{current_spot:.2f}", f"Last update: {timestamp}")
m2.metric("Confluence Score", score, "Action Threshold: ±2")
m3.metric("RSI (14)", f"{latest['RSI']:.2f}")
m4.metric("Active Pattern", pat_data['pattern'])

floating_pnl = 0.0
if st.session_state.active_trade:
    trade = st.session_state.active_trade
    pts = (current_spot - trade['entry']) if trade['type'] == 'CE' else (trade['entry'] - current_spot)
    floating_pnl = pts * 0.5 * 25
m5.metric("Total Est. P&L (INR)", f"₹{(st.session_state.total_pnl + floating_pnl):.2f}", f"Floating: ₹{floating_pnl:.2f}")

# --- Interactive Chart ---
st.subheader("Live Technical Analysis Chart")
fig = go.Figure()

# Candlesticks
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Nifty 50'))

# EMAs
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='blue', width=1), name='EMA 20'))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='orange', width=1), name='EMA 50'))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_100'], line=dict(color='red', width=1), name='EMA 100'))

fig.update_layout(height=500, template='plotly_dark', xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

# --- Trading Panel ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Position Status")
    if st.session_state.active_trade:
        tr = st.session_state.active_trade
        st.success(f"**ACTIVE: LONG {tr['type']}**")
        st.write(f"**Entry Spot:** {tr['entry']:.2f} at {tr['time']}")
        st.write(f"**Current Signals:** {', '.join(reasons)}")
    else:
        st.info("Market Neutral. Waiting for high-probability setup...")
        st.write(f"**Current Signals:** {', '.join(reasons) if reasons else 'None'}")

with col2:
    st.subheader("Paper Trade History")
    if st.session_state.trade_history:
        history_df = pd.DataFrame(st.session_state.trade_history)
        # Format PNL with color formatting
        st.dataframe(history_df.style.map(lambda x: 'color: green' if x > 0 else 'color: red', subset=['pnl']), use_container_width=True)
    else:
        st.write("No trades executed today.")

# Auto-refresh trigger (Loops the script every 60 seconds if checked)
if auto_refresh:
    time.sleep(60)
    st.rerun()