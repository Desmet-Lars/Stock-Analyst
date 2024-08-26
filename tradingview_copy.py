from ast import Global
import yfinance as yf
import pandas as pd
import threading
import logging
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
import time

# Initialize the Dash app
app = Dash(__name__)
server = app.server

# Time intervals
INTERVALS = {
    '1 Minute': '1m',
    '5 Minutes': '5m',
    '15 Minutes': '15m',
    '1 Hour': '1h',
    '1 Day': '1d',
    '1 Week': '1wk'
}

# Stock categories
STOCK_CATEGORIES = {
    'Crypto': ['BTC-USD', 'ETH-USD', 'BNB-USD'],
    'Crypto Miners': ['MARA', 'RIOT', 'HUT'],
    'Magnificent 7': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
}
MONITORED_STOCKS = ['BTC-USD', 'ETH-USD', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA','MARA', 'RIOT', 'HUT']

# Trading simulation variables
FAKE_BALANCE = 10000
STOCK_INVENTORY = {}
TRADES_LOG = []
TRADE_RESULT = {"message": ""}  # For notifications
TRADE_RESULT_LOCK = threading.Lock()

# Function to get stock data
def get_stock_data(symbol, interval='1d'):
    intraday_intervals = ['1m', '5m', '15m', '1h']
    max_days_intraday = 60
    max_days_historical = {
        '1d': 730, '5d': 730, '1wk': 730, '1mo': 730, '3mo': 730,
        '6mo': 730, '1y': 365, '2y': 730, '5y': 1825, '10y': 3650, 'ytd': 365
    }

    today = datetime.now()
    start_date = None
    end_date = today

    if interval in intraday_intervals:
        start_date = today - timedelta(days=max_days_intraday)
    else:
        days = max_days_historical.get(interval, 730)
        start_date = today - timedelta(days=days)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    try:
        df = yf.download(symbol, start=start_str, end=end_str, interval=interval, progress=False)
        if df.empty:
            raise ValueError("No data returned from Yahoo Finance.")
        
        df = df.reset_index()
        if 'Date' not in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol} with interval {interval}: {e}")
        return pd.DataFrame()

# Add indicators (MA, RSI, MACD, Stochastic RSI)
def add_indicators(df, indicators):
    if 'MA50' in indicators:
        df['MA50'] = df['Close'].rolling(window=50).mean()
    if 'MA100' in indicators:
        df['MA100'] = df['Close'].rolling(window=100).mean()
    if 'MA200' in indicators:
        df['MA200'] = df['Close'].rolling(window=200).mean()
    
    if 'RSI' in indicators:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    if 'MACD' in indicators:
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    if 'StochRSI' in indicators:
        rsi = df['RSI']
        stoch_rsi = (rsi - rsi.rolling(window=14).min()) / (rsi.rolling(window=14).max() - rsi.rolling(window=14).min())
        df['StochRSI'] = stoch_rsi * 100

    return df

# Detect parallel channels
def detect_channels(df):
    df['Channel_high'] = df['Close'].rolling(window=20).apply(lambda x: x.max(), raw=True)
    df['Channel_low'] = df['Close'].rolling(window=20).apply(lambda x: x.min(), raw=True)
    return df

# Forecast future price using SARIMA
def predict_future_price(df, days=30):
    df = df.copy()
    if 'Date' not in df.columns:
        raise ValueError("The DataFrame does not contain a 'Date' column.")
    
    df.set_index('Date', inplace=True)
    
    if df.empty:
        return pd.DataFrame(columns=['Date', 'Predicted_Price'])
    
    model = SARIMAX(df['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
    model_fit = model.fit(disp=False)
    
    future_dates = [df.index.max() + timedelta(days=i) for i in range(1, days + 1)]
    forecast = model_fit.get_forecast(steps=days).predicted_mean
    
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': forecast
    })

    return future_df


# Save trades log
def save_trade_log(stock, action, amount, price, date):
    global TRADES_LOG
    TRADES_LOG.append({
        'Stock': stock,
        'Action': action,
        'Amount': amount,
        'Price': price,
        'Date': date
    })

# Update stock inventory
import threading

# Create locks for thread-safe updates
balance_lock = threading.Lock()
inventory_lock = threading.Lock()

# Update stock inventory with thread safety
def update_inventory(stock, amount, action):
    global STOCK_INVENTORY
    with inventory_lock:
        if action == 'buy':
            if stock in STOCK_INVENTORY:
                STOCK_INVENTORY[stock] += amount
            else:
                STOCK_INVENTORY[stock] = amount
        elif action == 'sell':
            if stock in STOCK_INVENTORY and STOCK_INVENTORY[stock] >= amount:
                STOCK_INVENTORY[stock] -= amount
                if STOCK_INVENTORY[stock] == 0:
                    del STOCK_INVENTORY[stock]
            else:
                return False
    return True

# Simulate trade with thread safety for balance
def simulate_trade(stock, action, amount):
    global FAKE_BALANCE
    with balance_lock:
        df = get_stock_data(stock, '1d')
        if df.empty:
            return "Failed to fetch stock data."

        current_price = df['Close'].iloc[-1]
        if action == 'buy':
            total_cost = current_price * amount
            if FAKE_BALANCE >= total_cost:
                FAKE_BALANCE -= total_cost
                if update_inventory(stock, amount, 'buy'):
                    save_trade_log(stock, 'buy', amount, current_price, datetime.now())
                    return f"Bought {amount} of {stock} at ${current_price:.2f}. Remaining balance: ${FAKE_BALANCE:.2f}"
                else:
                    return "Failed to update inventory after buying."
            else:
                return "Insufficient balance."
        elif action == 'sell':
            if stock in STOCK_INVENTORY and STOCK_INVENTORY[stock] >= amount:
                total_revenue = current_price * amount
                FAKE_BALANCE += total_revenue
                if update_inventory(stock, amount, 'sell'):
                    save_trade_log(stock, 'sell', amount, current_price, datetime.now())
                    return f"Sold {amount} of {stock} at ${current_price:.2f}. New balance: ${FAKE_BALANCE:.2f}"
                else:
                    return "Failed to update inventory after selling."
            else:
                return "Insufficient stock to sell."



# Automated trading bot with prediction-based trading and quantity adjustment
def automated_trading_bot():
    global FAKE_BALANCE
    while True:
        for stock in MONITORED_STOCKS:
            try:
                # Fetch stock data
                df = get_stock_data(stock, '1d')
                if df.empty:
                    logging.warning(f"No data for {stock}")
                    continue

                # Add indicators and detect channels
                df = add_indicators(df, ['RSI', 'MACD'])
                df = detect_channels(df)

                # Predict future prices
                future_df = predict_future_price(df, days=30)
                current_price = df['Close'].iloc[-1]
                predicted_price = future_df['Predicted_Price'].iloc[-1]

                # Define thresholds and quantities
                buy_threshold = 0.04  # 4% increase for buying
                sell_threshold = 0.05  # 5% decrease for selling
                buy_fraction = 0.1  # 10% of available capital
                sell_fraction = 0.5  # 50% of inventory to sell if price goes down

                # Debug statements
                print(f"Stock: {stock}")
                print(f"Current Price: ${current_price:.2f}, Predicted Price: ${predicted_price:.2f}")

                threshold_price = current_price * (1 + buy_threshold)
                print(f"Buy Threshold Price: ${threshold_price:.2f}")

                # Initialize trade quantities
                buy_quantity = 0
                sell_quantity = 0

                # Trading decision based on prediction
                if predicted_price >= threshold_price:
                    # Buy based on a percentage of available capital
                    buy_quantity = int((FAKE_BALANCE * buy_fraction) / current_price)
                    print(f"Calculated Buy Quantity: {buy_quantity}")

                    if FAKE_BALANCE >= current_price * buy_quantity:
                        trade_result = simulate_trade(stock, 'buy', buy_quantity)
                        print(f"Trade Result: {trade_result}")

                        if 'Bought' in trade_result and buy_quantity > 0:
                            FAKE_BALANCE -= current_price * buy_quantity
                            STOCK_INVENTORY[stock] = STOCK_INVENTORY.get(stock, 0) + buy_quantity

                            # Record trade in history
                            with TRADE_RESULT_LOCK:
                                TRADE_HISTORY.append({
                                    'stock': stock,
                                    'action': 'buy',
                                    'quantity': buy_quantity,
                                    'price': current_price,
                                    'total_cost': current_price * buy_quantity
                                })
                    else:
                        print("Insufficient balance to buy.")
                else:
                    print("Buy condition not met.")

                # Handle selling
                if predicted_price <= current_price * (1 - sell_threshold):
                    if stock in STOCK_INVENTORY:
                        if predicted_price < current_price * (1 - sell_threshold):
                            # Sell all if price drops more than the threshold
                            sell_quantity = STOCK_INVENTORY.get(stock, 0)
                        else:
                            # Sell a fraction if price drops less than the threshold
                            sell_quantity = int(STOCK_INVENTORY.get(stock, 0) * sell_fraction)
                        
                        trade_result = simulate_trade(stock, 'sell', sell_quantity)
                        print(f"Trade Result: {trade_result}")

                        if 'Sold' in trade_result and sell_quantity > 0:
                            FAKE_BALANCE += current_price * sell_quantity
                            STOCK_INVENTORY[stock] -= sell_quantity
                            if STOCK_INVENTORY[stock] == 0:
                                del STOCK_INVENTORY[stock]

                            # Record trade in history
                            with TRADE_RESULT_LOCK:
                                TRADE_HISTORY.append({
                                    'stock': stock,
                                    'action': 'sell',
                                    'quantity': sell_quantity,
                                    'price': current_price,
                                    'total_revenue': current_price * sell_quantity
                                })
                    else:
                        print("No stock in inventory to sell.")
                else:
                    print("Sell condition not met.")

                # Update trade result
                with TRADE_RESULT_LOCK:
                    # Only update TRADE_RESULT if there was a trade action
                    if buy_quantity > 0 or sell_quantity > 0:
                        TRADE_RESULT["message"] = (f"Stock: {stock}, Current Price: ${current_price:.2f}, "
                                                   f"Predicted Price: ${predicted_price:.2f}, Buy Quantity: {buy_quantity}, "
                                                   f"Sell Quantity: {sell_quantity}")
                    else:
                        TRADE_RESULT["message"] = f"No action taken for Stock: {stock}"

                logging.info(TRADE_RESULT["message"])

            except Exception as e:
                logging.error(f"Error during trading for {stock}: {e}")

        time.sleep(10)  # Delay to avoid overwhelming the API



# Layout
app.layout = html.Div(style={'backgroundColor': '#121212', 'color': '#e0e0e0', 'height': '100vh', 'margin': '0', 'display': 'flex'}, children=[
    html.Div([
        html.H1("Stock Dashboard", style={'textAlign': 'center', 'color': '#f5f5f5'}),
        dcc.Tabs(id='category-tabs', value='Crypto', children=[
            dcc.Tab(label='Crypto', value='Crypto'),
            dcc.Tab(label='Crypto Miners', value='Crypto Miners'),
            dcc.Tab(label='Magnificent 7', value='Magnificent 7')
        ], style={'backgroundColor': '#1f1f1f', 'color': '#e0e0e0'}),
        dcc.Dropdown(
            id='stock-dropdown',
            style={'backgroundColor': '#1f1f1f', 'color': '#e0e0e0', 'marginBottom': '20px'}
        ),
        dcc.Dropdown(
            id='interval-dropdown',
            options=[{'label': label, 'value': value} for label, value in INTERVALS.items()],
            value='1d',
            style={'backgroundColor': '#1f1f1f', 'color': '#e0e0e0', 'marginBottom': '20px'}
        ),
        dcc.Checklist(
            id='indicator-checklist',
            options=[
                {'label': 'MA50', 'value': 'MA50'},
                {'label': 'MA100', 'value': 'MA100'},
                {'label': 'MA200', 'value': 'MA200'},
                {'label': 'RSI', 'value': 'RSI'},
                {'label': 'MACD', 'value': 'MACD'},
                {'label': 'Stochastic RSI', 'value': 'StochRSI'}
            ],
            value=['RSI', 'MACD'],
            style={'backgroundColor': '#1f1f1f', 'color': '#e0e0e0', 'marginBottom': '20px'}
        ),
        html.Button('Run Automated Trading Bot', id='run-bot-button', n_clicks=0, style={'marginBottom': '20px', 'backgroundColor': '#007bff', 'color': '#fff', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer'}),
        html.Div(id='trade-result', style={'color': '#e0e0e0', 'marginTop': '20px'}),
        html.H2("Portfolio", style={'marginTop': '20px'}),
        html.Div(id='portfolio', style={'color': '#e0e0e0', 'marginBottom': '20px'}),
        html.H2("Trade History", style={'marginTop': '20px'}),
        html.Div(id='trade-history', style={'color': '#e0e0e0'})
    ], style={'width': '25%', 'backgroundColor': '#1f1f1f', 'padding': '20px', 'boxSizing': 'border-box'}),
    
    html.Div([
        dcc.Graph(id='stock-graph', config={'displayModeBar': True, 'scrollZoom': True}),
        html.Div(id='prediction-text', style={'padding': '20px', 'color': '#e0e0e0'})
    ], style={'width': '75%', 'height': '100%', 'padding': '20px', 'boxSizing': 'border-box'})
])

# Update stock dropdown based on selected category
@app.callback(
    Output('stock-dropdown', 'options'),
    Input('category-tabs', 'value')
)
def update_stock_dropdown(selected_category):
    stocks = STOCK_CATEGORIES.get(selected_category, [])
    return [{'label': stock, 'value': stock} for stock in stocks]

# Update graph, prediction, portfolio, and trade history

@app.callback(
    [Output('stock-graph', 'figure'),
     Output('prediction-text', 'children'),
     Output('portfolio', 'children'),
     Output('trade-history', 'children')],
    [Input('stock-dropdown', 'value'),
     Input('interval-dropdown', 'value'),
     Input('indicator-checklist', 'value')]
)
def update_dashboard(selected_stock, selected_interval, selected_indicators):
    if not selected_stock:
        return go.Figure(), "Select a stock to see predictions.", "", ""

    df = get_stock_data(selected_stock, selected_interval)
    df = add_indicators(df, selected_indicators)
    df = detect_channels(df)
    future_df = predict_future_price(df, days=30)
    
    # Create the figure with dynamic subplot layout
    rows = 2 if selected_indicators else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.3 if selected_indicators else 0.2)
    
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Candlestick'
    ), row=1, col=1)

    if selected_indicators:
        for indicator in selected_indicators:
            if indicator.startswith('MA'):
                fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator), row=1, col=1)
        
        if 'RSI' in selected_indicators:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI'), row=2, col=1)
        
        if 'MACD' in selected_indicators:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_signal'], mode='lines', name='MACD Signal'), row=2, col=1)
        
        if 'StochRSI' in selected_indicators:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['StochRSI'], mode='lines', name='Stochastic RSI'), row=2, col=1)

    fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted_Price'], mode='lines', name='Forecast'), row=1, col=1)

    fig.update_layout(
        template='plotly_dark', title=f'{selected_stock} Price Analysis',
        xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False,
        height=800 if selected_indicators else 600
    )

    next_date = future_df['Date'].iloc[-1].strftime('%Y-%m-%d')
    next_price = future_df['Predicted_Price'].iloc[-1]
    prediction_text = f"Predicted Price on {next_date}: ${next_price:.2f}"

    # Provide explanations based on selected indicators
    explanations = []
    if 'RSI' in selected_indicators:
        if df['RSI'].iloc[-1] > 70:
            explanations.append("RSI indicates the stock is overbought.")
        elif df['RSI'].iloc[-1] < 30:
            explanations.append("RSI indicates the stock is oversold.")
    
    if 'MACD' in selected_indicators:
        if df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]:
            explanations.append("MACD line is above the MACD signal line.")
        else:
            explanations.append("MACD line is below the MACD signal line.")

    if 'StochRSI' in selected_indicators:
        if df['StochRSI'].iloc[-1] > 0.8:
            explanations.append("Stochastic RSI is in the overbought region.")
        elif df['StochRSI'].iloc[-1] < 0.2:
            explanations.append("Stochastic RSI is in the oversold region.")

    explanation_text = " ".join(explanations) if explanations else "No significant indicators."

    # Update portfolio information
    portfolio_info = f"Current balance: ${FAKE_BALANCE:.2f}\n"
    for stock, quantity in STOCK_INVENTORY.items():
        portfolio_info += f"{stock}: {quantity} shares\n"
    
    trade_history_info = f"Last trade: {TRADE_RESULT['message']}" if TRADE_RESULT.get("message") else "No trades yet."

    return fig, prediction_text, portfolio_info, trade_history_info



# Run the automated trading bot in a separate thread
def run_bot():
    threading.Thread(target=automated_trading_bot, daemon=True).start()

# Start the trading bot when the button is clicked
@app.callback(
    Output('trade-result', 'children'),
    Input('run-bot-button', 'n_clicks')
)
def run_trading_bot(n_clicks):
    if n_clicks > 0:
        run_bot()
        return "Trading bot is now running. Check the results for updates."
    return ""

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
