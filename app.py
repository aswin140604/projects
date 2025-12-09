import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

# Set page config
st.set_page_config(
    page_title="MetaTrader 5 Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .profit {
        color: green;
    }
    .loss {
        color: red;
    }
    .title {
        text-align: center;
        font-size: 2rem;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit app title
st.markdown("<h1 class='title'>MetaTrader 5 Trading Dashboard</h1>", unsafe_allow_html=True)

# Function to initialize MT5 connection
def initialize_mt5(account_number, password, server):
    if not mt5.initialize():
        st.error("MT5 initialization failed")
        return False
    
    if not mt5.login(account_number, password=password, server=server):
        st.error("Login failed! Check your credentials.")
        mt5.shutdown()
        return False
    else:
        st.success(f"Successfully logged into MT5 Account: {account_number}")
        return True

# Function to get account info
def get_account_info():
    account_info = mt5.account_info()
    if account_info is not None:
        return {
            "Balance": account_info.balance,
            "Equity": account_info.equity,
            "Margin": account_info.margin,
            "Free Margin": account_info.margin_free,
            "Margin Level": account_info.margin_level if account_info.margin > 0 else 0,
            "Profit": account_info.profit
        }
    return None

# Function to get open trades
def get_open_trades():
    open_trades = mt5.positions_get()
    if open_trades:
        data = []
        for trade in open_trades:
            data.append({
                "Symbol": trade.symbol,
                "Type": "Buy" if trade.type == 0 else "Sell",
                "Volume": trade.volume,
                "Open Price": trade.price_open,
                "Current Price": trade.price_current,
                "Profit": trade.profit,
                "SL": trade.sl,
                "TP": trade.tp
            })
        return pd.DataFrame(data)
    return pd.DataFrame()

# Function to get closed trades
def get_closed_trades():
    closed_trades = mt5.history_deals_get(datetime.now() - timedelta(days=30), datetime.now())
    if closed_trades:
        data = []
        for trade in closed_trades:
            data.append({
                "Symbol": trade.symbol,
                "Type": "Buy" if trade.type == 0 else "Sell",
                "Volume": trade.volume,
                "Price": trade.price,
                "Profit": trade.profit,
                "Time": pd.to_datetime(trade.time, unit='s')
            })
        return pd.DataFrame(data)
    return pd.DataFrame()

# Function to fetch candle data
def fetch_candle_data(symbol, timeframe, bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        st.error(f"Failed to retrieve {symbol} data")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index("time", inplace=True)
    return df

# Function to calculate ATR
def calculate_atr(symbol, timeframe, period=14):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + 1)
    if rates is None or len(rates) == 0:
        st.error("Failed to retrieve data for ATR calculation")
        return 0
        
    df = pd.DataFrame(rates)
    
    # Calculate True Range
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_close'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df.dropna(inplace=True)
    
    # Calculate ATR
    atr = df['tr'].mean()
    return atr

# Function to train ML model
@st.cache_resource
def train_ml_model(symbol, timeframe, bars=5000):
    df = fetch_candle_data(symbol, timeframe, bars)
    if df is None:
        return None, None, None, None, None, None

    # Feature engineering
    df['returns'] = df['close'].pct_change().fillna(0)
    df['target'] = df['close'].shift(-1)
    df.dropna(inplace=True)

    X = df[['open', 'high', 'low', 'close', 'tick_volume']]
    y = df['target'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train).flatten()
    y_test_scaled = scaler_y.transform(y_test).flatten()

    # Build Neural Network
    model = keras.Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(), 
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    lr_schedule = ExponentialDecay(initial_learning_rate=0.01, decay_steps=1000, decay_rate=0.9)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss='huber_loss', metrics=['mae'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    
    with st.spinner('Training model... This may take a few minutes.'):
        model.fit(
            X_train_scaled, y_train_scaled, 
            epochs=100, batch_size=64, 
            validation_data=(X_test_scaled, y_test_scaled),
            callbacks=[early_stopping, reduce_lr], 
            verbose=0
        )

    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_actual = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_test_actual, y_pred)
    r2 = r2_score(y_test_actual, y_pred)

    return model, mse, r2, X_test, y_test_actual, y_pred

def make_predictions(model, symbol, timeframe, scaler_X, scaler_y):
    df = fetch_candle_data(symbol, timeframe, 10)
    if df is not None and model is not None:
        last_row = df.iloc[-1][['open', 'high', 'low', 'close', 'tick_volume']].values.reshape(1, -1)
        last_row_scaled = scaler_X.transform(last_row)
        predicted_price_scaled = model.predict(last_row_scaled)[0]
        
        if hasattr(predicted_price_scaled, '__len__'):
            predicted_price_scaled = predicted_price_scaled[0]
        predicted_price = scaler_y.inverse_transform([[predicted_price_scaled]])[0][0]
        current_price = df.iloc[-1]['close']
        
        return current_price, predicted_price
    return None, None

# Function to plot candles using Plotly
def plot_candles_plotly(df, title):
    if df is None:
        return None
    
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, 
                         vertical_spacing=0.03, subplot_titles=[title])
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Candlestick"
        ),
        row=1, col=1
    )
    
    # Layout update
    fig.update_layout(
        height=500,
        margin=dict(t=50, b=50, l=50, r=50),
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )
    
    return fig

# Function to plot model performance
def plot_model_performance(actual, predicted):
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=list(range(len(actual))),
        y=actual,
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))
    
    # Add predicted values
    fig.add_trace(go.Scatter(
        x=list(range(len(predicted))),
        y=predicted,
        mode='lines',
        name='Predicted',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title="Model Prediction vs Actual",
        xaxis_title="Data Point Index",
        yaxis_title="Price",
        height=500,
        template="plotly_white"
    )
    
    return fig

# Sidebar - Login to MT5
with st.sidebar:
    st.header("📊 MT5 Connection")
    with st.form("login_form"):
        account_number = st.number_input("Account Number", min_value=1, value=12345678)
        password = st.text_input("Password", type="password", value="Your_password")
        server = st.text_input("Server", value="Exness-MT5Trial7")
        symbol = st.text_input("Symbol", value="BTCUSDm")
        
        # Timeframe selection
        timeframe_options = {
            "4 Hours": mt5.TIMEFRAME_H4,
        }
        selected_timeframe = st.selectbox("Timeframe", list(timeframe_options.keys()))
        timeframe = timeframe_options[selected_timeframe]
        
        # Number of bars
        bars = st.slider("Number of bars", min_value=50, max_value=2000, value=500)
        
        # Submit button
        submit_button = st.form_submit_button("Connect to MT5")
    
    if submit_button or 'mt5_connected' in st.session_state:
        if 'mt5_connected' not in st.session_state:
            st.session_state.mt5_connected = initialize_mt5(account_number, password, server)
            if st.session_state.mt5_connected:
                st.session_state.symbol = symbol
                st.session_state.timeframe = timeframe
                st.session_state.bars = bars
        
        if st.session_state.mt5_connected:
            # Additional sidebar options
            st.header("🔄 Refresh Data")
            if st.button("Refresh Dashboard"):
                st.experimental_rerun()
            
            # Model training option
            st.header("🧠 Model Training")
            if st.button("Train ML Model"):
                st.session_state.train_model = True

# Main content area
if 'mt5_connected' in st.session_state and st.session_state.mt5_connected:
    # Create tabs
    tabs = st.tabs(["📊 Market Overview", "💰 Account Info", "🤖 ML Predictions"])
    
    # Tab 1: Market Overview
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{st.session_state.symbol} ({selected_timeframe})")
            df = fetch_candle_data(st.session_state.symbol, st.session_state.timeframe, st.session_state.bars)
            if df is not None:
                fig = plot_candles_plotly(df, f"{st.session_state.symbol} Chart")
                st.plotly_chart(fig, use_container_width=True)
                
                # Display latest price data
                latest = df.iloc[-1]
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.subheader("Latest Price Data")
                latest_cols = st.columns(4)
                latest_cols[0].metric("Open", f"${latest['open']:.2f}")
                latest_cols[1].metric("High", f"${latest['high']:.2f}")
                latest_cols[2].metric("Low", f"${latest['low']:.2f}")
                latest_cols[3].metric("Close", f"${latest['close']:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # ATR
                atr = calculate_atr(st.session_state.symbol, st.session_state.timeframe)
                st.metric("Average True Range (ATR)", f"{atr:.4f}")
        
        with col2:
            # Open trades
            st.subheader("Running Trades")
            open_trades_df = get_open_trades()
            if not open_trades_df.empty:
                st.dataframe(open_trades_df, use_container_width=True)
            else:
                st.info("No running trades")
            
            # Closed trades
            st.subheader("Recent Closed Trades (Last 30 days)")
            closed_trades_df = get_closed_trades()
            if not closed_trades_df.empty:
                st.dataframe(closed_trades_df, use_container_width=True)
                
                # Trades summary
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.subheader("Trading Summary")
                trades_summary_cols = st.columns(4)
                total_profit = closed_trades_df['Profit'].sum()
                profit_class = "profit" if total_profit >= 0 else "loss"
                trades_summary_cols[0].metric("Total Trades", len(closed_trades_df))
                trades_summary_cols[1].metric("Total Profit/Loss", f"${total_profit:.2f}")
                trades_summary_cols[2].metric("Win Trades", len(closed_trades_df[closed_trades_df['Profit'] > 0]))
                trades_summary_cols[3].metric("Loss Trades", len(closed_trades_df[closed_trades_df['Profit'] <= 0]))
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No closed trades in the last 30 days")
    
    # Tab 2: Account Info
    with tabs[1]:
        # Account information
        account_info = get_account_info()
        if account_info:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.subheader("Account Overview")
            account_cols = st.columns(3)
            account_cols[0].metric("Balance", f"${account_info['Balance']:.2f}")
            account_cols[1].metric("Equity", f"${account_info['Equity']:.2f}")
            
            profit_class = "profit" if account_info['Profit'] >= 0 else "loss"
            account_cols[2].metric("Profit/Loss", f"${account_info['Profit']:.2f}")
            
            margin_cols = st.columns(3)
            margin_cols[0].metric("Margin", f"${account_info['Margin']:.2f}")
            margin_cols[1].metric("Free Margin", f"${account_info['Free Margin']:.2f}")
            margin_cols[2].metric("Margin Level", f"{account_info['Margin Level']:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.subheader("Equity Curve")
        closed_trades_df = get_closed_trades()
        if not closed_trades_df.empty:
            closed_trades_df = closed_trades_df.sort_values('Time')
            equity_data = closed_trades_df['Profit'].cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=closed_trades_df['Time'],
                y=equity_data + account_info['Balance'] - equity_data.iloc[-1],
                mode='lines',
                name='Equity',
                line=dict(color='green')
            ))
            
            fig.update_layout(
                title="Account Equity Over Time",
                xaxis_title="Date",
                yaxis_title="Equity ($)",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trade history available to generate equity curve")
    
    with tabs[2]:
        # Train model button already in sidebar
        if 'train_model' in st.session_state and st.session_state.train_model:
            # Train model
            model, mse, r2, X_test, y_test, y_pred = train_ml_model(
                st.session_state.symbol, 
                st.session_state.timeframe, 
                5000  # Use more data for training
            )
            
            # Store model and results in session state
            st.session_state.model = model
            st.session_state.mse = mse
            st.session_state.r2 = r2
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            
            st.session_state.train_model = False
        
        if 'model' in st.session_state:
            col1, col2 = st.columns(2)
            
            with col1:
                # Display model performance metrics
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.subheader("Model Performance Metrics")
                metrics_cols = st.columns(2)
                metrics_cols[0].metric("Mean Squared Error (MSE)", f"{st.session_state.mse:.6f}")
                metrics_cols[1].metric("R² Score", f"{st.session_state.r2:.6f}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Plot actual vs predicted
                fig = plot_model_performance(st.session_state.y_test, st.session_state.y_pred)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Next price prediction
                current_price, next_price = make_predictions(
                    st.session_state.model, 
                    st.session_state.symbol, 
                    st.session_state.timeframe,
                    MinMaxScaler().fit(st.session_state.X_test),
                    MinMaxScaler().fit(st.session_state.y_test.reshape(-1, 1))
                )
                
                if current_price is not None and next_price is not None:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.subheader("Price Prediction")
                    pred_cols = st.columns(3)
                    pred_cols[0].metric("Current Price", f"${current_price:.2f}")
                    
                    price_change = next_price - current_price
                    price_change_pct = (price_change / current_price) * 100
                    trend = "↑" if price_change > 0 else "↓"
                    
                    pred_cols[1].metric(
                        "Predicted Next Price", 
                        f"${next_price:.2f}", 
                        f"{trend} {price_change:.2f} ({price_change_pct:.2f}%)"
                    )
                    
                    signal = "BUY" if price_change > 0 else "SELL"
                    pred_cols[2].metric("Signal", signal)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Trading recommendation
                    st.subheader("Trading Recommendation")
                    if abs(price_change_pct) < 0.1:
                        st.info("Price change is within noise threshold. No clear signal.")
                    else:
                        if price_change > 0:
                            st.success(f"Consider BUYING {st.session_state.symbol} - Model predicts price increase of {price_change_pct:.2f}%")
                        else:
                            st.error(f"Consider SELLING {st.session_state.symbol} - Model predicts price decrease of {abs(price_change_pct):.2f}%")
                        
                        # Show ATR for stop loss/take profit suggestions
                        atr = calculate_atr(st.session_state.symbol, st.session_state.timeframe)
                        if atr > 0:
                            if price_change > 0:  # Buy recommendation
                                sl = current_price - (atr * 1.5)
                                tp = current_price + (atr * 1.5 * 2)  # 1:2 risk-reward
                            else:  # Sell recommendation
                                sl = current_price + (atr * 1.5)
                                tp = current_price - (atr * 1.5 * 2)  # 1:2 risk-reward
                            
                            st.markdown(f"**Suggested Entry:** ${current_price:.2f}")
                            st.markdown(f"**Suggested Stop Loss:** ${sl:.2f} ({(sl-current_price)/current_price*100:.2f}%)")
                            st.markdown(f"**Suggested Take Profit:** ${tp:.2f} ({(tp-current_price)/current_price*100:.2f}%)")
                            
                            # Execute trade button
                            if st.button("Execute Trade"):
                                execute_trade(st.session_state.symbol, signal, current_price, sl, tp)
                                st.success(f"Trade executed: {signal} {st.session_state.symbol} at ${current_price:.2f}")
        else:
            st.warning("Please train the ML model using the button in the sidebar")

else:
    # Show landing page when not connected
    st.image("https://mql5.com/assets/icons/mt5-brand2020.svg", width=100)
    st.markdown("""
    ## Welcome to the MT5 Trading Dashboard
    
    This application connects to your MetaTrader 5 account and provides:
    
    - 📊 **Real-time market data visualization**
    - 💰 **Account information monitoring**
    - 📈 **Trading history analysis**
    - 🤖 **ML-based price predictions**
    
    To get started, please enter your MT5 account details in the sidebar and click "Connect to MT5".
    """)

# Disconnect from MT5 when app is closed
def disconnect_mt5():
    if mt5.initialize():
        mt5.shutdown()

import atexit
atexit.register(disconnect_mt5)
