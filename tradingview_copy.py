import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

# Function to get stock data
def get_stock_data(symbol, interval='1d'):
    df = yf.download(symbol, period='1y', interval=interval, progress=False)
    df = df.dropna().reset_index()  # Remove missing data and reset index
    if 'USD' not in symbol:  # Non-crypto stocks
        df = df[df['Date'].dt.dayofweek < 5]  # Remove weekends
    return df

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
    df['Channel_high'] = df['Close'].rolling(window=20).apply(lambda x: x.max())
    df['Channel_low'] = df['Close'].rolling(window=20).apply(lambda x: x.min())
    
    return df

# Forecast future price using SARIMA
def predict_future_price(df, days=30):
    df = df.copy()
    df.set_index('Date', inplace=True)
    
    # SARIMA model
    model = SARIMAX(df['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
    model_fit = model.fit()
    
    future_dates = [df.index.max() + timedelta(days=i) for i in range(1, days + 1)]
    forecast = model_fit.get_forecast(steps=days).predicted_mean
    
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': forecast
    })

    return future_df

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
        html.Label("Indicators:", style={'color': '#e0e0e0'}),
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
            value=['RSI', 'MACD'],  # Default indicators
            style={'backgroundColor': '#1f1f1f', 'color': '#e0e0e0', 'marginBottom': '20px'}
        )
    ], style={'width': '20%', 'backgroundColor': '#1f1f1f', 'padding': '20px', 'boxSizing': 'border-box'}),
    
    html.Div([
        dcc.Graph(id='stock-graph', config={'displayModeBar': True, 'scrollZoom': True}),
        html.Div(id='prediction-text', style={'padding': '20px', 'color': '#e0e0e0'})
    ], style={'width': '80%', 'height': '100%', 'padding': '20px', 'boxSizing': 'border-box'})
])

# Update stock dropdown based on selected category
@app.callback(
    Output('stock-dropdown', 'options'),
    Input('category-tabs', 'value')
)
def update_stock_dropdown(selected_category):
    return [{'label': stock, 'value': stock} for stock in STOCK_CATEGORIES[selected_category]]

# Callback to update the graph
@app.callback(
    [Output('stock-graph', 'figure'),
     Output('prediction-text', 'children')],
    [Input('stock-dropdown', 'value'),
     Input('interval-dropdown', 'value'),
     Input('indicator-checklist', 'value')]
)
def update_graph(selected_stock, selected_interval, selected_indicators):
    if not selected_stock:
        return {}, "Select a stock to see predictions."

    df = get_stock_data(selected_stock, selected_interval)
    df = add_indicators(df, selected_indicators)
    df = detect_channels(df)
    
    # Predict future price
    future_df = predict_future_price(df, days=30)
    
    # Determine number of rows for the subplot
    num_rows = 1
    if 'RSI' in selected_indicators:
        num_rows += 1
    if 'MACD' in selected_indicators:
        num_rows += 1
    if 'StochRSI' in selected_indicators:
        num_rows += 1

    fig = make_subplots(
        rows=num_rows, cols=1, shared_xaxes=True,
        row_heights=[0.7] + [0.3] * (num_rows - 1),
        vertical_spacing=0.05
    )

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Candlestick'), row=1, col=1)
    
    # Add indicators
    if 'MA50' in selected_indicators:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], mode='lines', name='MA50', line=dict(color='#e377c2')), row=1, col=1)
    if 'MA100' in selected_indicators:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA100'], mode='lines', name='MA100', line=dict(color='#7f7f7f')), row=1, col=1)
    if 'MA200' in selected_indicators:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA200'], mode='lines', name='MA200', line=dict(color='#bcbd22')), row=1, col=1)
    
    if 'MACD' in selected_indicators:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD', line=dict(color='#8c564b')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_signal'], mode='lines', name='MACD Signal', line=dict(color='#17becf')), row=2, col=1)

    if 'StochRSI' in selected_indicators:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['StochRSI'], mode='lines', name='Stochastic RSI', line=dict(color='#00bfae')), row=3, col=1)
    
    # Add parallel channel lines
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Channel_high'], mode='lines', name='Channel High', line=dict(color='#FF5733', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Channel_low'], mode='lines', name='Channel Low', line=dict(color='#33FF57', dash='dash')), row=1, col=1)
    
    # Add prediction line
    fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted_Price'], mode='lines', name='Future Price', line=dict(color='#FF00FF')), row=1, col=1)

    # Update layout
    fig.update_layout(
        template='plotly_dark', title=f'{selected_stock} Price Analysis',
        xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False,
        height=800
    )
    
    # Prepare prediction text
    next_date = future_df['Date'].iloc[-1].strftime('%Y-%m-%d')
    next_price = future_df['Predicted_Price'].iloc[-1]
    prediction_text = f"Predicted Price on {next_date}: ${next_price:.2f}"

    # Explanation for price movement
    explanation = []
    if 'RSI' in selected_indicators:
        if df['RSI'].iloc[-1] > 70:
            explanation.append("RSI indicates the stock is overbought, suggesting a potential price drop.")
        elif df['RSI'].iloc[-1] < 30:
            explanation.append("RSI indicates the stock is oversold, suggesting a potential price increase.")
    
    if 'MACD' in selected_indicators:
        if df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]:
            explanation.append("MACD line is above the MACD signal line, indicating bullish momentum.")
        else:
            explanation.append("MACD line is below the MACD signal line, indicating bearish momentum.")

    if 'StochRSI' in selected_indicators:
        if df['StochRSI'].iloc[-1] > 80:
            explanation.append("Stochastic RSI indicates the stock is overbought.")
        elif df['StochRSI'].iloc[-1] < 20:
            explanation.append("Stochastic RSI indicates the stock is oversold.")
    
    if explanation:
        prediction_text += "\n".join(explanation)

    return fig, prediction_text

if __name__ == '__main__':
    app.run_server(debug=True)
