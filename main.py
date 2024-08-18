import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from dash.exceptions import PreventUpdate
from sklearn.linear_model import LinearRegression
import logging
from plotly.subplots import make_subplots
from flask_caching import Cache

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define popular stocks
POPULAR_STOCKS = ['BTC-USD', 'LULU', 'MARA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

# Initialize Dash app
app = Dash(__name__)
server = app.server

# Set up Flask-Caching
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

# Define layout with modern dark mode styling and responsive design
app.layout = html.Div(style={
    'fontFamily': 'Arial, sans-serif',
    'margin': '0',
    'padding': '0',
    'backgroundColor': '#121212',  # Dark background
    'color': '#e0e0e0',
    'boxSizing': 'border-box',
    'overflowX': 'hidden'
}, children=[
    html.Header([
        html.H1("Stock Dashboard", style={
            'textAlign': 'center',
            'margin': '20px 0',
            'color': '#f5f5f5',
            'fontSize': '36px',
            'fontWeight': '700',
            'letterSpacing': '1px'
        }),
    ], style={
        'backgroundColor': '#1f1f1f',
        'padding': '20px',
        'boxShadow': '0 2px 10px rgba(0, 0, 0, 0.3)'
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
        'boxSizing': 'border-box',
        'maxWidth': '1400px',
        'margin': '0 auto'
    })
])

def get_stock_data(symbol, interval='1m'):
    end_time = pd.Timestamp.now()
    start_time = end_time - pd.Timedelta(days=1)
    
    try:
        df = yf.download(symbol, start=start_time, end=end_time, interval=interval, progress=False)
        if df.empty:
            logging.warning(f"No data returned for {symbol}. MARKET CLOSED ")
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        
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
    if df.empty:
        return None, None, None, None, None, None, None
    
    try:
        df['change'] = df['Close'].diff()
        df['gain'] = df['change'].clip(lower=0)
        df['loss'] = -df['change'].clip(upper=0)
        
        avg_gain = df['gain'].rolling(window=14, min_periods=1).mean()
        avg_loss = df['loss'].rolling(window=14, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        initial_close = df['Close'].iloc[0]
        latest_close = df['Close'].iloc[-1]
        percent_change = ((latest_close - initial_close) / initial_close) * 100
        
        rsi_oversold = df['RSI'].iloc[-1] < 30  # RSI oversold if below 30
        
        df['time'] = np.arange(len(df))
        df['hour_of_day'] = df.index.hour + df.index.minute / 60.0
        
        features = df[['time', 'hour_of_day', 'Volume']]
        X = features.values
        y = df['Close'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        future_times = np.arange(len(df), len(df) + 10).reshape(-1, 1)
        future_hours = (df.index[-1] + pd.to_timedelta(np.arange(1, 11), unit='T')).hour + \
                        (df.index[-1] + pd.to_timedelta(np.arange(1, 11), unit='T')).minute / 60.0
        future_volumes = df['Volume'].rolling(window=5).mean().iloc[-1]
        
        future_features = np.column_stack([future_times, future_hours, [future_volumes]*10])
        future_prices = model.predict(future_features)
        
        future_percent_change = ((future_prices[-1] - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100
        
        prediction_text = f"""
        Detailed Prediction:\n
        - Expected percentage change over the next 10 intervals: {future_percent_change:.2f}%\n
        - Expected price after 10 intervals: ${future_prices[-1]:.2f}\n
        - Expected volatility in the next 10 intervals: {np.std(future_prices):.2f}\n
        - Predicted price range in the next 10 intervals: ${min(future_prices):.2f} - ${max(future_prices):.2f}
        """
        
        recommendation = "HOLD"
        if percent_change < -5:
            recommendation = "SELL"
        elif percent_change > 5:
            recommendation = "BUY"
        
        return percent_change, rsi_oversold, df, recommendation, future_times.flatten(), future_prices, prediction_text
    except Exception as e:
        logging.error(f"Error analyzing data: {e}")
        return None, None, None, None, None, None, None

def create_candlestick_chart(df, symbol, candlestick_colors=None, chart_height=800, future_times=None, future_prices=None):
    try:
        if candlestick_colors is None:
            candlestick_colors = {
                'bullish_up': '#76ff03',
                'bearish_down': '#ff1744',
                'bullish_fill': 'rgba(118, 255, 3, 0.2)',
                'bearish_fill': 'rgba(255, 23, 68, 0.2)'
            }
        
        fig = make_subplots(rows=1, cols=1)  # Single subplot for both candlestick and RSI

        # Candlestick chart
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
        ))

        # RSI line on the same chart
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='#ff9100')
        ))

        # Future price prediction line
        if future_times is not None and future_prices is not None:
            future_dates = df.index[-1] + pd.to_timedelta(np.arange(1, len(future_prices) + 1), unit='T')
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_prices,
                mode='lines+markers',
                name='Predicted Prices',
                line=dict(color='#00e5ff', dash='dash')
            ))
        
        fig.update_layout(
            title=f'{symbol} Candlestick Chart with RSI',
            xaxis_title='Date',
            yaxis_title='Price/RSI',
            height=chart_height,
            template='plotly_dark',
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(
                title="Price",
                side="right"
            ),
            yaxis2=dict(
                title="RSI",
                overlaying="y",
                side="left",
                range=[0, 100]
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig
    except Exception as e:
        logging.error(f"Error creating chart for {symbol}: {e}")
        return None

@app.callback(
    Output('dashboard-content', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_dashboard(n_intervals):
    logging.info(f"Callback triggered with n_intervals: {n_intervals}")
    if n_intervals is None:
        raise PreventUpdate

    reports = []

    for symbol in POPULAR_STOCKS:
        try:
            df = get_stock_data(symbol)
            logging.info(f"Data fetched for {symbol}: {df.shape}")
            
            percent_change, rsi_oversold, df_with_rsi, recommendation, future_times, future_prices, prediction_text = analyze_stock(df)
            logging.info(f"Analysis completed for {symbol}")
            
            if df.empty:
                reports.append(html.Div(f"Data for {symbol} not available. MARKET CLOSED ", style={'color': '#ff1744', 'fontSize': '18px'}))
                continue
            
            candlestick_colors = {
                'bullish_up': '#76ff03',
                'bearish_down': '#ff1744',
                'bullish_fill': 'rgba(118, 255, 3, 0.2)',
                'bearish_fill': 'rgba(255, 23, 68, 0.2)'
            }
            chart_height = 700
            
            fig = create_candlestick_chart(df_with_rsi, symbol, candlestick_colors=candlestick_colors, chart_height=chart_height, future_times=future_times, future_prices=future_prices)
            if fig is None:
                continue
            
            graph = dcc.Graph(
                figure=fig,
                style={'height': f'{chart_height}px', 'width': '100%', 'margin': '0 auto'}
            )
            
            report = html.Div([
                html.H2(symbol, style={'textAlign': 'center', 'color': '#e1e1e1'}),
                graph,
                html.Div([
                    html.P(f"Current percent change: {percent_change:.2f}%"),
                    html.P(f"RSI Oversold: {'Yes' if rsi_oversold else 'No'}"),
                    html.P(f"Recommendation: {recommendation}"),
                    html.Pre(prediction_text, style={'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace'})
                ], style={'textAlign': 'left', 'color': '#e1e1e1', 'fontSize': '16px'})
            ], style={
                'width': '90%', 'maxWidth': '1200px', 'backgroundColor': '#1f1f1f',
                'padding': '20px', 'marginBottom': '20px', 'borderRadius': '10px',
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.4)'
            })
            
            reports.append(report)
        
        except Exception as e:
            logging.error(f"Error processing data for {symbol}: {e}")
            reports.append(html.Div(f"Error processing {symbol}: {e}", style={'color': '#ff1744'}))
    
    logging.info(f"Reports generated: {len(reports)}")
    return reports if reports else [html.Div("No data available.", style={'color': '#ff1744'})]

if __name__ == '__main__':
    app.run_server(debug=True)
