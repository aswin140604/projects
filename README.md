# 📊 Multi Asset Price Prediction Using MT5

## 🚀 Overview

This MetaTrader 5 Trading Dashboard is a powerful Streamlit application that provides comprehensive trading analytics, real-time market insights, and machine learning-powered price predictions.

![Demo Website](images/1.png)

## ✨ Features

### 1. Market Overview 📈
- Real-time candlestick charts
- Latest price data
- Average True Range (ATR) calculation
- Running and closed trades tracking
![Demo Website](images/7.png)
![Demo Website](images/2_1.png)
![Demo Website](images/2_2.png)


### 2. Account Information 💰
- Balance and equity tracking
- Margin details
- Profit/Loss metrics
- Equity curve visualization
![Demo Website](images/3.png)



### 3. Machine Learning Predictions 🤖
- Neural network-based price prediction
- Model performance metrics
- Trading signal recommendations
- Stop loss and take profit suggestions
![Demo Website](images/4.jpeg)

## 🛠 Prerequisites

### Required Libraries
- streamlit
- pandas
- numpy
- matplotlib
- mplfinance
- MetaTrader5
- scikit-learn
- tensorflow
- plotly

### Python Version
- Python 3.8+

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/Hemanth-310/MetaTrader.git
cd MetaTrader
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 🚦 Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. In the sidebar, enter your MetaTrader 5 credentials:
- Account Number
- Password
- Server
- Symbol (e.g., BTCUSDm, XAUUSDm)
- Timeframe
- Number of bars to analyze

3. Click "Connect to MT5"

## 🔬 Machine Learning Model

### Model Architecture
- Input Layer: 128 neurons with ReLU activation
- Batch Normalization
- Dropout (20%)
- Hidden Layers: 64 and 32 neurons with ReLU activation
- Output Layer: Single neuron for price prediction

```python
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
```
### Training Process
- Uses Exponential Learning Rate Decay
- Early Stopping
- Reduce Learning Rate on Plateau
- Huber Loss Function
- Mean Absolute Error Metric
  
```python
    lr_schedule = ExponentialDecay(initial_learning_rate=0.01, decay_steps=1000, decay_rate=0.9)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='huber_loss', metrics=['mae'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    
    # Train the Model
    model.fit(X_train, y_train, epochs=500, batch_size=64, validation_data=(X_test, y_test),
              callbacks=[early_stopping, reduce_lr], verbose=1)
```
## 📊 Performance Metrics

The ML model provides:
- Mean Squared Error (MSE)
- R² Score
- Actual vs. Predicted Price Comparison
- Trading Signals (Buy/Sell)
![Demo Website](images/5.png)
![Demo Website](images/6.png)

## 🔒 Security Notes

- Securely handle MT5 credentials
- Implement additional authentication if required
- Use environment variables for sensitive information

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## ⚠️ Disclaimer

Trading involves financial risk. This dashboard is for educational and analytical purposes. Always conduct your own research and consult financial advisors before making trading decisions.


