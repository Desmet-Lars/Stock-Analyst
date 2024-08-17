import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from dash.exceptions import PreventUpdate
from sklearn.linear_model import LinearRegression
import logging
from plotly.subplots import make_subplots  # Add this import


# Set up logging
logging.basicConfig(level=logging.INFO)

# Define popular stocks
POPULAR_STOCKS = ['BTC-USD', 'LULU', 'MARA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

# Initialize Dash app
app = Dash(__name__)

# Define layout with modern dark mode styling and responsive design
app.layout = html.Div(style={
    'fontFamily': 'Arial, sans-serif',
    'margin': '0',
    'padding': '0',
    'backgroundColor': '#1e1e1e',
    'color': '#e1e1e1',
    'boxSizing': 'border-box',
    'overflowX': 'hidden'
}, children=[
    html.H1("Stock Dashboard", style={
        'textAlign': 'center',
        'margin': '20px',
        'color': '#e1e1e1',
        'fontSize': '36px',
        'fontWeight': '700'
    }),
    dcc.Interval(
        id='interval-component',
        interval=1*60*1000,  # Update every 1 minute
        n_intervals=0
    ),
    html.Div(id='dashboard-content', style={
        'padding': '20px',
        'display': 'flex',
        'flexWrap': 'wrap',
        'justifyContent': 'center',
        'boxSizing': 'border-box'
    })
])

def get_stock_data(symbol, interval='1m'):
    """
    Fetches intraday stock data for the last 24 hours from Yahoo Finance.
    """
    end_time = pd.Timestamp.now()
    start_time = end_time - pd.Timedelta(days=1)
    
    try:
        df = yf.download(symbol, start=start_time, end=end_time, interval=interval, progress=False)
        if df.empty:
            logging.warning(f"No data returned for {symbol}.")
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        
        # Resample data to a higher time frame (e.g., 5-minute)
        df = df.resample('5T').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def analyze_stock(df):
    """
    Analyzes the stock data to check for crashes, RSI, and provides a detailed prediction.
    """
    if df.empty:
        return None, None, None, None, None, None, None
    
    try:
        # Calculate RSI
        df['change'] = df['Close'].diff()
        df['gain'] = df['change'].clip(lower=0)
        df['loss'] = -df['change'].clip(upper=0)
        
        avg_gain = df['gain'].rolling(window=14, min_periods=1).mean()
        avg_loss = df['loss'].rolling(window=14, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Check for significant crash
        initial_close = df['Close'].iloc[0]
        latest_close = df['Close'].iloc[-1]
        percent_change = ((latest_close - initial_close) / initial_close) * 100
        
        rsi_oversold = df['RSI'].iloc[-1] < 30  # RSI oversold if below 30
        
        # Enhanced Linear Regression Prediction with Volume
        df['time'] = np.arange(len(df))
        df['hour_of_day'] = df.index.hour + df.index.minute / 60.0
        
        features = df[['time', 'hour_of_day', 'Volume']]
        X = features.values  # Convert to numpy array
        y = df['Close'].values  # Convert to numpy array
        
        model = LinearRegression()
        model.fit(X, y)
        
        future_times = np.arange(len(df), len(df) + 10).reshape(-1, 1)
        future_hours = (df.index[-1] + pd.to_timedelta(np.arange(1, 11), unit='T')).hour + \
                        (df.index[-1] + pd.to_timedelta(np.arange(1, 11), unit='T')).minute / 60.0
        future_volumes = df['Volume'].rolling(window=5).mean().iloc[-1]  # Predict future volumes as moving average
        
        future_features = np.column_stack([future_times, future_hours, [future_volumes]*10])
        future_prices = model.predict(future_features)
        
        # Calculate percentage increase over the next 10 intervals
        future_percent_change = ((future_prices[-1] - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100
        
        prediction_text = f"""
        Detailed Prediction:\n
        - Expected percentage change over the next 10 intervals: {future_percent_change:.2f}%\n
        - Expected price after 10 intervals: ${future_prices[-1]:.2f}\n
        - Expected volatility in the next 10 intervals: {np.std(future_prices):.2f}\n
        - Predicted price range in the next 10 intervals: ${min(future_prices):.2f} - ${max(future_prices):.2f}
        """
        
        # Basic recommendation logic
        if percent_change < -5:
            recommendation = "SELL"
        elif percent_change > 5:
            recommendation = "BUY"
        else:
            recommendation = "HOLD"
        
        return percent_change, rsi_oversold, df, recommendation, future_times.flatten(), future_prices, prediction_text
    except Exception as e:
        logging.error(f"Error analyzing data: {e}")
        return None, None, None, None, None, None, None


def create_candlestick_chart(df, symbol, candlestick_colors=None, chart_height=800, future_times=None, future_prices=None):
    """
    Creates a customizable candlestick chart for the stock data with a separate RSI chart.
    """
    try:
        # Default colors
        if candlestick_colors is None:
            candlestick_colors = {
                'bullish_up': 'lime',
                'bearish_down': 'red',
                'bullish_fill': 'rgba(0, 255, 0, 0.2)',
                'bearish_fill': 'rgba(255, 0, 0, 0.2)'
            }
        
        # Create subplots with 2 rows and shared x-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05,  # Adjust spacing between plots
                            row_heights=[0.7, 0.3])  # Adjust height ratio

        # Add candlestick chart to the first row
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlestick',
            increasing_line_color=candlestick_colors['bullish_up'],
            decreasing_line_color=candlestick_colors['bearish_down'],
            increasing_fillcolor=candlestick_colors['bullish_fill'],
            decreasing_fillcolor=candlestick_colors['bearish_fill']
        ), row=1, col=1)

        # Add future price predictions if provided
        if future_times is not None and future_prices is not None:
            future_dates = df.index[-1] + pd.to_timedelta(np.arange(1, len(future_prices) + 1), unit='T')
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_prices,
                mode='lines+markers',
                name='Predicted Prices',
                line=dict(color='cyan', dash='dash')
            ), row=1, col=1)
        
        # Add RSI line to the second row
        if 'RSI' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple')
            ), row=2, col=1)
        
        # Add horizontal lines to indicate RSI thresholds (70 and 30)
        fig.add_hline(y=70, line=dict(color='red', dash='dash'), row=2, col=1)
        fig.add_hline(y=30, line=dict(color='green', dash='dash'), row=2, col=1)

        # Update layout
        fig.update_layout(
            title=f'{symbol} Candlestick Chart with RSI',
            xaxis_title='Date',
            yaxis_title='Price',
            yaxis2_title='RSI',
            template='plotly_dark',  # Use dark mode template
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font=dict(color='#e1e1e1'),
            margin=dict(l=0, r=0, t=50, b=0),
            height=chart_height,  # Overall chart height
            xaxis_rangeslider_visible=False  # Range slider will be included on RSI chart
        )

        # Update second y-axis (RSI) layout specifically
        fig.update_yaxes(range=[0, 100], row=2, col=1)  # Set RSI y-axis range from 0 to 100

        return fig
    except Exception as e:
        logging.error(f"Error creating chart for {symbol}: {e}")
        return go.Figure()


@app.callback(
    Output('dashboard-content', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_dashboard(n_intervals):
    if n_intervals is None:
        raise PreventUpdate
    
    reports = []
    
    for symbol in POPULAR_STOCKS:
        df = get_stock_data(symbol)
        percent_change, rsi_oversold, df_with_rsi, recommendation, future_times, future_prices, prediction_text = analyze_stock(df)
        
        if df.empty:
            reports.append(html.Div(f"Data for {symbol} not available.", style={'color': '#e1e1e1', 'fontSize': '18px'}))
            continue

        # Color-code the crash status based on the percentage change
        if percent_change is not None:
            if abs(percent_change) > 5:
                crash_status = f"CRASH: {percent_change:.2f}%"
                crash_color = 'red' if percent_change < 0 else 'lime'
            else:
                crash_status = f"Change: {percent_change:.2f}%"
                crash_color = 'red' if percent_change < 0 else 'lime'
        else:
            crash_status = "No significant change"
            crash_color = 'grey'
        
        rsi_status = "RSI Oversold" if rsi_oversold else "RSI Normal"
        
        reports.append(html.Div([
            html.Div([
                html.H3(symbol, style={'textAlign': 'center', 'color': '#e1e1e1', 'fontSize': '24px', 'fontWeight': '600'}),
                dcc.Loading(
                    id=f'loading-{symbol}',
                    children=[
                        dcc.Graph(
                            id=f'graph-{symbol}',
                            figure=create_candlestick_chart(df_with_rsi, symbol, 
                                candlestick_colors={
                                    'bullish_up': 'cyan',
                                    'bearish_down': 'orange',
                                    'bullish_fill': 'rgba(0, 255, 255, 0.2)',
                                    'bearish_fill': 'rgba(255, 165, 0, 0.2)'
                                },
                                chart_height=800,
                                future_times=future_times,
                                future_prices=future_prices
                            ), 
                            style={'height': '800px', 'width': '100%'}
                        )
                    ],
                    type='circle'  # Type of loading spinner
                ),
                html.P(crash_status, style={'fontSize': '20px', 'fontWeight': 'bold', 'color': crash_color}),
                html.P(rsi_status, style={'fontSize': '18px', 'color': '#e1e1e1'}),
                html.P(f"Recommendation: {recommendation}", style={'fontSize': '18px', 'fontWeight': 'bold', 'color': '#e1e1e1'}),
                html.P(prediction_text, style={'fontSize': '18px', 'color': '#e1e1e1'})
            ], style={'marginBottom': '20px', 'padding': '20px', 'backgroundColor': '#2a2a2a', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.3)', 'width': '100%', 'maxWidth': '1200px'})
        ], style={'width': '100%', 'maxWidth': '1200px', 'margin': '0 auto'}))
    
    return reports

if __name__ == '__main__':
    app.run_server(debug=True)
