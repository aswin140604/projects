import MetaTrader5 as mt5
import pandas as pd
import mplfinance as mpf
import time
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import statistics
import threading

# MT5 Account Credentials
ACCOUNT_NUMBER = 12345678
PASSWORD = "your_password"
SERVER = "Exness-MT5Trial7"
SYMBOL = "BTCUSDm"

# Connect to MT5
if not mt5.initialize():
    print("❌ MT5 initialization failed")
    quit()

# Login to MT5
if not mt5.login(ACCOUNT_NUMBER, password=PASSWORD, server=SERVER):
    print("❌ Login failed! Check your credentials.")
    mt5.shutdown()
    quit()
else:
    print(f"✅ Successfully logged into MT5 Account: {ACCOUNT_NUMBER}")

# Fetch Account Information
def get_account_info():
    account_info = mt5.account_info()
    if account_info is not None:
        print(f"💰 Balance: {account_info.balance}, Equity: {account_info.equity}, Margin: {account_info.margin}")

# Fetch Open Trades
def get_open_trades():
    open_trades = mt5.positions_get()
    if open_trades:
        print("\n📌 Running Trades:")
        for trade in open_trades:
            print(f"🔹 {trade.symbol} | {trade.type} | Volume: {trade.volume} | Price: {trade.price_open}")
    else:
        print("✅ No running trades")

# Fetch Closed Trades
def get_closed_trades():
    closed_trades = mt5.history_deals_get(datetime(2024, 1, 1), datetime.now())
    if closed_trades:
        print("\n📌 Closed Trades:")
        for trade in closed_trades[-5:]:  # Last 5 closed trades
            print(f"🔻 {trade.symbol} | {trade.type} | Volume: {trade.volume} | Price: {trade.price}")
    else:
        print("✅ No closed trades")

# Fetch Candlestick Data
def fetch_candle_data(symbol, timeframe, bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        print(f"❌ Failed to retrieve {symbol} data")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index("time", inplace=True)
    return df

# Plot Candlestick Chart
def plot_candles(df, title):
    if df is not None:
        mpf.plot(df, type="candle", style="charles", volume=False,
                 title=title, ylabel="Price (USD)", figsize=(12, 6))

# Helper function to calculate ATR (Average True Range)
def calculate_atr(symbol, timeframe, period=14):
    """Calculate Average True Range for dynamic stop loss/take profit"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + 1)
    if rates is None or len(rates) == 0:
        print("❌ Failed to retrieve data for ATR calculation")
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

# Calculate Risk Reward Ratio
def calculate_risk_reward(entry_price, stop_loss, take_profit, order_type):
    """Calculate risk-to-reward ratio for a trade"""
    if order_type == "buy":
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
    else:  # sell
        risk = stop_loss - entry_price
        reward = entry_price - take_profit
    
    rr_ratio = reward / risk if risk != 0 else 0
    return rr_ratio

# Calculate Win Rate From Past Hour
def calculate_win_rate(symbol, timeframe=mt5.TIMEFRAME_M1, min_win_rate=33.33):
    """Analyze past hour's data to determine win rate for our strategy"""
    # Get past hour's data
    current_time = datetime.now()
    past_hour = current_time - timedelta(hours=1)
    
    # Fetch historical data for the past hour
    rates = mt5.copy_rates_range(symbol, timeframe, past_hour, current_time)
    if rates is None or len(rates) == 0:
        print("❌ Failed to retrieve historical data for win rate calculation")
        return False
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Simple strategy simulation for backtesting
    df['signal'] = 0  # 1 for buy, -1 for sell
    
    # Use a simple model for win rate calculation
    df['returns'] = df['close'].pct_change().fillna(0)
    df['signal'] = np.where(df['returns'] > 0, 1, -1)  # Buy if price increases, sell if decreases
    
    # Calculate wins and losses
    df['next_close'] = df['close'].shift(-1)
    df['profit'] = np.where(df['signal'] == 1, 
                           df['next_close'] - df['close'],
                           df['close'] - df['next_close'])
    
    # Count wins
    df.dropna(inplace=True)
    total_trades = len(df)
    if total_trades == 0:
        print("❌ Not enough data to calculate win rate")
        return False
        
    winning_trades = len(df[df['profit'] > 0])
    
    win_rate = (winning_trades / total_trades) * 100
    print(f"🎯 Past Hour Win Rate: {win_rate:.2f}% (from {total_trades} trades)")
    return win_rate >= min_win_rate

# Fetch and Plot Candlestick Charts
plot_candles(fetch_candle_data(SYMBOL, mt5.TIMEFRAME_M1, 1440), "1-Minute (Last 1 Day)")
plot_candles(fetch_candle_data(SYMBOL, mt5.TIMEFRAME_D1, 30), "1-Day (Last 1 Month)")
plot_candles(fetch_candle_data(SYMBOL, mt5.TIMEFRAME_W1, 12), "1-Week (Last 3 Months)")
plot_candles(fetch_candle_data(SYMBOL, mt5.TIMEFRAME_MN1, 12), "1-Month (Last 1 Year)")

# Machine Learning Model
def train_ml_model():
    df = fetch_candle_data(SYMBOL, mt5.TIMEFRAME_M1, 5000)
    if df is None:
        return None, None, None, None

    df['returns'] = df['close'].pct_change().fillna(0)
    df['target'] = df['close'].shift(-1)
    df.dropna(inplace=True)

    X = df[['open', 'high', 'low', 'close', 'tick_volume']]
    y = df['target'].values.reshape(-1, 1)  # Reshape y for scaling

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fix MinMaxScaler error & properly scale both X & y
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train).flatten()
    y_test = scaler_y.transform(y_test).flatten()

    # Optimized Neural Network
    model = keras.Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # Use Learning Rate Decay
    lr_schedule = ExponentialDecay(initial_learning_rate=0.01, decay_steps=1000, decay_rate=0.9)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    # Use Huber Loss (robust to outliers)
    model.compile(optimizer=optimizer, loss='huber_loss', metrics=['mae'])

    # Callbacks: Early Stopping & Reduce LR on Plateau
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    
    # Train the Model
    model.fit(X_train, y_train, epochs=500, batch_size=64, validation_data=(X_test, y_test),
              callbacks=[early_stopping, reduce_lr], verbose=1)

    # Predict & Inverse Transform y_pred
    y_pred = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📈 Optimized NN Model Trained!")
    print(f"🔹 R² Score: {r2:.4f}")
    print(f"🔻 MSE: {mse:.4f}")

    return model, X_train, scaler_X, scaler_y  # Return scalers for later use

# Backtesting
def backtest(model, X):
    if model is None:
        print("❌ No trained model found. Skipping backtest.")
        return

    if isinstance(X, np.ndarray):  # Convert to DataFrame if necessary
        X = pd.DataFrame(X)

    predictions = model.predict(X[-100:])  # Last 100 predictions
    actuals = range(len(predictions))  # Use index range instead

    df_results = pd.DataFrame({"Time": actuals, "Predicted": predictions.flatten()})
    print("\n📊 Backtesting Results (Last 10 Predictions):")
    print(df_results.tail(10))

# Place Trade Manually (Original)
def place_trade(order_type, lot_size=0.01, slippage=2):
    price = mt5.symbol_info_tick(SYMBOL).ask if order_type == "buy" else mt5.symbol_info_tick(SYMBOL).bid
    order_type_mt5 = mt5.ORDER_TYPE_BUY if order_type == "buy" else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot_size,
        "type": order_type_mt5,
        "price": price,
        "deviation": slippage,
        "magic": 0,
        "comment": "Manual Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"✅ Trade Executed: {order_type.upper()} {lot_size} lots at {price}")
    else:
        print(f"❌ Trade Failed: {result.comment}")

# Enhanced Place Trade with Risk Management (1:2 risk-reward)
def place_trade_with_risk_management(order_type, lot_size=0.01, risk_reward_ratio=2.0, slippage=2):
    """Place trade with proper risk management (1:2 risk-reward ratio)"""
    # Get current price
    tick = mt5.symbol_info_tick(SYMBOL)
    
    if order_type == "buy":
        entry_price = tick.ask
        # Calculate stop loss and take profit based on ATR
        atr = calculate_atr(SYMBOL, mt5.TIMEFRAME_M5, 14)
        stop_loss = entry_price - (atr * 1.5)  # 1.5 times ATR for stop loss
        take_profit = entry_price + (atr * 1.5 * risk_reward_ratio)  # 1:2 ratio
        
        order_type_mt5 = mt5.ORDER_TYPE_BUY
    else:  # sell
        entry_price = tick.bid
        atr = calculate_atr(SYMBOL, mt5.TIMEFRAME_M5, 14)
        stop_loss = entry_price + (atr * 1.5)
        take_profit = entry_price - (atr * 1.5 * risk_reward_ratio)
        
        order_type_mt5 = mt5.ORDER_TYPE_SELL
    
    # Calculate actual risk-reward ratio
    actual_rr = calculate_risk_reward(entry_price, stop_loss, take_profit, order_type)
    print(f"📊 Trade Risk:Reward = 1:{actual_rr:.2f}")
    
    if actual_rr < risk_reward_ratio * 0.9:  # Allow some flexibility (90% of target)
        print(f"❌ Risk-reward ratio too low ({actual_rr:.2f}). Trade not executed.")
        return False
    
    # Create the order request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot_size,
        "type": order_type_mt5,
        "price": entry_price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": slippage,
        "magic": 0,
        "comment": f"Auto Trade RR=1:{actual_rr:.1f}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # Send the order
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"✅ Trade Executed: {order_type.upper()} {lot_size} lots at {entry_price}")
        print(f"   Stop Loss: {stop_loss}, Take Profit: {take_profit}")
        return True
    else:
        print(f"❌ Trade Failed: {result.comment} (code: {result.retcode})")
        return False

# Enhanced Automatic Trading with Win Rate Check and Risk Management
def auto_trade():
    model, X, scaler_X, scaler_y = train_ml_model()
    backtest(model, X)

    while True:
        # Check win rate before placing new trades
        win_rate_sufficient = calculate_win_rate(SYMBOL, mt5.TIMEFRAME_M1, 33.33)
        
        if not win_rate_sufficient:
            print("⚠️ Win rate below threshold (33.33%). Not taking new trades this cycle.")
            get_account_info()
            get_open_trades()
            get_closed_trades()
            print("\n⏳ Waiting for next minute...")
            time.sleep(60)
            continue
            
        df = fetch_candle_data(SYMBOL, mt5.TIMEFRAME_M1, 10)
        if df is not None and model is not None:
            last_row = df.iloc[-1][['open', 'high', 'low', 'close', 'tick_volume']].values.reshape(1, -1)
            last_row_scaled = scaler_X.transform(last_row)
            predicted_price = model.predict(last_row_scaled)[0]
            
            # Convert prediction back to actual price
            if hasattr(predicted_price, '__len__'):
                predicted_price = predicted_price[0]  # Handle different model output shapes
            predicted_price = scaler_y.inverse_transform([[predicted_price]])[0][0]
            current_price = df.iloc[-1]['close']

            print(f"Current price: {current_price}, Predicted next price: {predicted_price}")

            if predicted_price > current_price * 1.001:  # Add 0.1% threshold to reduce false signals
                print("📢 Predicted price is higher! Placing BUY order...")
                place_trade_with_risk_management("buy", risk_reward_ratio=2.0)
            elif predicted_price < current_price * 0.999:  # Add 0.1% threshold to reduce false signals
                print("📢 Predicted price is lower! Placing SELL order...")
                place_trade_with_risk_management("sell", risk_reward_ratio=2.0)
            else:
                print("⚠️ Prediction within noise threshold. No trade.")

        get_account_info()
        get_open_trades()
        get_closed_trades()

        print("\n⏳ Waiting for next minute...")
        time.sleep(60)

# Run auto-trading in background
trading_thread = threading.Thread(target=auto_trade, daemon=True)
trading_thread.start()

# Enhanced Manual Trade Option
while True:
    action = input("\nEnter 'buy' or 'sell' to place trade, 'simple' for simple trade without risk management, or 'exit' to quit: ").strip().lower()
    
    if action == "buy" or action == "sell":
        # Check win rate before manual trade
        win_rate_sufficient = calculate_win_rate(SYMBOL, mt5.TIMEFRAME_M1, 33.33)
        if not win_rate_sufficient:
            print("⚠️ Warning: Win rate below 33.33% in the past hour")
            confirm = input("Do you still want to proceed with the trade? (y/n): ").strip().lower()
            if confirm != 'y':
                continue
                
        # Get lot size
        try:
            lot_size = float(input("Enter lot size (default: 0.01): ") or "0.01")
        except ValueError:
            lot_size = 0.01
            
        # Get risk:reward ratio
        try:
            rr_ratio = float(input("Enter risk:reward ratio (default: 2.0): ") or "2.0")
        except ValueError:
            rr_ratio = 2.0
            
        place_trade_with_risk_management(action, lot_size, rr_ratio)
    
    elif action == "simple":
        order_type = input("Enter 'buy' or 'sell' for simple trade: ").strip().lower()
        if order_type in ["buy", "sell"]:
            try:
                lot_size = float(input("Enter lot size (default: 0.01): ") or "0.01")
            except ValueError:
                lot_size = 0.01
            place_trade(order_type, lot_size)
    
    elif action == "exit":
        print("🛑 Exiting manual trading.")
        break

# ✅ Disconnect from MT5
mt5.shutdown()
print("🔌 Disconnected from MT5")

