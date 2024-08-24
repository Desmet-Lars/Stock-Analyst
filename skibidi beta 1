import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define popular stocks and indicators
POPULAR_STOCKS = ['BTC-USD', 'LULU', 'MARA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
INDICATORS = ['RSI', 'MACD', 'Moving Average']

# Initialize Dash app
app = Dash(__name__)
server = app.server

# Define layout
app.layout = html.Div(style={
    'fontFamily': 'Arial, sans-serif',
    'backgroundColor': '#121212',
    'color': '#e0e0e0',
    'boxSizing': 'border-box',
    'overflowX': 'hidden'
}, children=[
    html.Header([
        html.H1("Stock Dashboard", style={
            'textAlign': 'center',
            'color': '#f5f5f5',
            'fontSize': '24px',
            'fontWeight': '700',
            'letterSpacing': '1px'
        }),
    ], style={
        'backgroundColor': '#1f1f1f',
        'padding': '10px',
        'boxShadow': '0 2px 10px rgba(0, 0, 0, 0.3)'
    }),
    html.Div([
        dcc.Dropdown(
            id='stock-dropdown',
            options=[{'label': stock, 'value': stock} for stock in POPULAR_STOCKS],
            value='BTC-USD',
            style={'backgroundColor': '#1f1f1f', 'color': '#e0e0e0', 'marginBottom': '20px', 'border': '1px solid #3d3d3d'}
        ),
        dcc.Dropdown(
            id='indicator-dropdown',
            options=[{'label': indicator, 'value': indicator} for indicator in INDICATORS],
            value=['RSI'],
            multi=True,
            style={'backgroundColor': '#1f1f1f', 'color': '#e0e0e0', 'marginBottom': '20px', 'border': '1px solid #3d3d3d'}
        )
    ], style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '0 20px'}),
    dcc.Interval(
        id='interval-component',
        interval=1*60*1000,
        n_intervals=0
    ),
    dcc.Graph(id='stock-graph', config={
        'displayModeBar': True,
        'scrollZoom': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': [
            'drawline', 'drawopenpath', 'drawrect', 'drawcircle', 'eraseshape'
        ],
        'editable': True
    }, style={'height': '800px', 'width': '100%', 'margin': '0 auto'}),
])

def get_stock_data(symbol, interval='1m'):
    end_time = pd.Timestamp.now()
    start_time = end_time - pd.Timedelta(days=1)
    
    df = yf.download(symbol, start=start_time, end=end_time, interval=interval, progress=False)
    if df.empty:
        logging.warning(f"No data returned for {symbol}. MARKET CLOSED")
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

def calculate_indicators(df, selected_indicators):
    indicators_data = {}
    if 'RSI' in selected_indicators:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        indicators_data['RSI'] = df['RSI']
    
    if 'MACD' in selected_indicators:
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        indicators_data['MACD'] = df[['MACD', 'Signal']]
    
    if 'Moving Average' in selected_indicators:
        df['MA20'] = df['Close'].rolling(window=20).mean()
        indicators_data['Moving Average'] = df['MA20']
    
    return indicators_data

def create_candlestick_chart(df, symbol, indicators_data):
    # Create subplots: Candlestick chart + chosen indicators
    fig = make_subplots(
        rows=1 + len(indicators_data),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.5] + [0.5 / len(indicators_data)] * len(indicators_data)
    )

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    ), row=1, col=1)

    # Plot each selected indicator in its own subplot
    for i, (indicator_name, indicator_data) in enumerate(indicators_data.items(), start=2):
        if indicator_name == 'RSI':
            fig.add_trace(go.Scatter(
                x=df.index,
                y=indicator_data,
                name='RSI',
                line=dict(color='#ff9100')
            ), row=i, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=i, col=1)
        elif indicator_name == 'MACD':
            fig.add_trace(go.Scatter(
                x=df.index,
                y=indicator_data['MACD'],
                name='MACD',
                line=dict(color='#00e5ff')
            ), row=i, col=1)
            fig.add_trace(go.Scatter(
                x=df.index,
                y=indicator_data['Signal'],
                name='Signal',
                line=dict(color='#ff1744')
            ), row=i, col=1)
            fig.update_yaxes(title_text="MACD", row=i, col=1)
        elif indicator_name == 'Moving Average':
            fig.add_trace(go.Scatter(
                x=df.index,
                y=indicator_data,
                name='MA20',
                line=dict(color='#76ff03')
            ), row=1, col=1)

    fig.update_layout(
        title=f'{symbol} Candlestick Chart',
        xaxis_title='Time',
        yaxis_title='Price',
        height=800,
        margin=dict(l=20, r=20, t=50, b=20),
        template='plotly_dark',
        showlegend=False,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        dragmode='drawline'
    )

    return fig

@app.callback(
    Output('stock-graph', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('stock-dropdown', 'value'),
     Input('indicator-dropdown', 'value')]
)
def update_dashboard(n_intervals, selected_stock, selected_indicators):
    logging.info(f"Callback triggered with n_intervals: {n_intervals}")
    if n_intervals is None or not selected_stock:
        raise PreventUpdate

    df = get_stock_data(selected_stock)
    if df.empty:
        return {}

    indicators_data = calculate_indicators(df, selected_indicators)
    fig = create_candlestick_chart(df, selected_stock, indicators_data)
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
