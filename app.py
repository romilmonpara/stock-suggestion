import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io

# Set page configuration
st.set_page_config(
    page_title="Stock Trading Suggestion System",
    page_icon="📈",
    layout="wide"
)

def get_data(ticker, start_date, end_date):
    """
    Fetch stock data using yfinance
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (datetime): Start date for data fetch
        end_date (datetime): End date for data fetch
    
    Returns:
        pd.DataFrame: Stock data with OHLCV columns
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def calculate_sma(data, window):
    """Calculate Simple Moving Average"""
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data, window=14):
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        data (pd.DataFrame): Stock data
        window (int): RSI calculation window
    
    Returns:
        pd.Series: RSI values
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        data (pd.DataFrame): Stock data
        fast (int): Fast EMA period
        slow (int): Slow EMA period
        signal (int): Signal line EMA period
    
    Returns:
        tuple: MACD line, Signal line, Histogram
    """
    ema_fast = data['Close'].ewm(span=fast).mean()
    ema_slow = data['Close'].ewm(span=slow).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_indicators(data):
    """
    Calculate all technical indicators
    
    Args:
        data (pd.DataFrame): Stock data
    
    Returns:
        pd.DataFrame: Data with added indicators
    """
    if data is None or data.empty:
        return None
    
    # Calculate indicators
    data['SMA_20'] = calculate_sma(data, 20)
    data['SMA_50'] = calculate_sma(data, 50)
    data['RSI'] = calculate_rsi(data)
    
    macd_line, signal_line, histogram = calculate_macd(data)
    data['MACD'] = macd_line
    data['MACD_Signal'] = signal_line
    data['MACD_Histogram'] = histogram
    
    return data

def generate_signals(data):
    """
    Generate Buy/Sell/Hold signals based on technical indicators
    
    Signal Logic:
    - Buy: SMA-20 > SMA-50 and RSI < 70
    - Sell: SMA-20 < SMA-50 and RSI > 30
    - Hold: Otherwise
    
    Args:
        data (pd.DataFrame): Data with indicators
    
    Returns:
        pd.DataFrame: Data with signals and confidence
    """
    if data is None or data.empty:
        return None
    
    signals = []
    confidence_scores = []
    
    for i in range(len(data)):
        sma_20 = data['SMA_20'].iloc[i]
        sma_50 = data['SMA_50'].iloc[i]
        rsi = data['RSI'].iloc[i]
        macd = data['MACD'].iloc[i]
        macd_signal = data['MACD_Signal'].iloc[i]
        
        # Skip if any indicator is NaN
        if pd.isna(sma_20) or pd.isna(sma_50) or pd.isna(rsi) or pd.isna(macd) or pd.isna(macd_signal):
            signals.append('Hold')
            confidence_scores.append(0)
            continue
        
        # Calculate signal based on conditions
        signal = 'Hold'
        confidence = 0
        
        # Buy conditions
        if sma_20 > sma_50 and rsi < 70:
            signal = 'Buy'
            # Calculate confidence based on indicator strength
            sma_strength = min((sma_20 - sma_50) / sma_50 * 100, 20)  # Cap at 20%
            rsi_strength = (70 - rsi) / 70 * 100
            macd_strength = 50 if macd > macd_signal else 0
            confidence = min((sma_strength + rsi_strength + macd_strength) / 3, 100)
        
        # Sell conditions
        elif sma_20 < sma_50 and rsi > 30:
            signal = 'Sell'
            # Calculate confidence based on indicator strength
            sma_strength = min((sma_50 - sma_20) / sma_50 * 100, 20)  # Cap at 20%
            rsi_strength = (rsi - 30) / 70 * 100
            macd_strength = 50 if macd < macd_signal else 0
            confidence = min((sma_strength + rsi_strength + macd_strength) / 3, 100)
        
        signals.append(signal)
        confidence_scores.append(max(0, confidence))
    
    data['Signal'] = signals
    data['Confidence'] = confidence_scores
    
    return data

def plot_charts(data, ticker):
    """
    Create interactive plots for stock data and indicators
    
    Args:
        data (pd.DataFrame): Data with indicators and signals
        ticker (str): Stock ticker symbol
    
    Returns:
        plotly.graph_objects.Figure: Interactive chart
    """
    if data is None or data.empty:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(f'{ticker} Stock Price with Moving Averages', 'RSI', 'MACD'),
        vertical_spacing=0.08,
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Stock price and moving averages
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='orange')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='red')),
        row=1, col=1
    )
    
    # Add buy/sell signals
    buy_signals = data[data['Signal'] == 'Buy']
    sell_signals = data[data['Signal'] == 'Sell']
    
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index, 
                y=buy_signals['Close'],
                mode='markers',
                name='Buy Signal',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ),
            row=1, col=1
        )
    
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['Close'],
                mode='markers',
                name='Sell Signal',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ),
            row=1, col=1
        )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    
    # Add RSI threshold lines as traces
    fig.add_trace(
        go.Scatter(x=data.index, y=[70]*len(data), mode='lines', 
                  name='RSI Overbought (70)', line=dict(color='red', dash='dash')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=[30]*len(data), mode='lines', 
                  name='RSI Oversold (30)', line=dict(color='green', dash='dash')),
        row=2, col=1
    )
    
    # MACD
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MACD_Signal'], name='MACD Signal', line=dict(color='red')),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Bar(x=data.index, y=data['MACD_Histogram'], name='MACD Histogram', marker_color='gray'),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"Stock Analysis Dashboard - {ticker}",
        title_x=0.5
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig

def simulate_backtest(data, initial_capital=10000):
    """
    Simulate a simple backtest based on generated signals
    
    Args:
        data (pd.DataFrame): Data with signals
        initial_capital (float): Starting capital
    
    Returns:
        pd.DataFrame: Backtest results with cumulative returns
    """
    if data is None or data.empty:
        return None
    
    # Initialize backtest variables
    position = 0  # 0 = no position, 1 = long position
    cash = initial_capital
    shares = 0
    portfolio_value = []
    
    for i in range(len(data)):
        signal = data['Signal'].iloc[i]
        price = data['Close'].iloc[i]
        
        if pd.isna(price):
            portfolio_value.append(cash + shares * (price if not pd.isna(price) else 0))
            continue
        
        # Execute trades based on signals
        if signal == 'Buy' and position == 0:
            # Buy with all available cash
            shares = cash / price
            cash = 0
            position = 1
        elif signal == 'Sell' and position == 1:
            # Sell all shares
            cash = shares * price
            shares = 0
            position = 0
        
        # Calculate current portfolio value
        current_value = cash + shares * price
        portfolio_value.append(current_value)
    
    # Create backtest results DataFrame
    backtest_data = data.copy()
    backtest_data['Portfolio_Value'] = portfolio_value
    backtest_data['Returns'] = (backtest_data['Portfolio_Value'] / initial_capital - 1) * 100
    
    return backtest_data

def create_backtest_chart(backtest_data, ticker):
    """Create backtest performance chart"""
    if backtest_data is None or backtest_data.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=backtest_data.index,
            y=backtest_data['Returns'],
            name='Strategy Returns (%)',
            line=dict(color='green')
        )
    )
    
    # Calculate buy and hold returns
    buy_hold_returns = (backtest_data['Close'] / backtest_data['Close'].iloc[0] - 1) * 100
    
    fig.add_trace(
        go.Scatter(
            x=backtest_data.index,
            y=buy_hold_returns,
            name='Buy & Hold Returns (%)',
            line=dict(color='blue')
        )
    )
    
    fig.update_layout(
        title=f'Backtest Performance Comparison - {ticker}',
        xaxis_title='Date',
        yaxis_title='Returns (%)',
        height=400
    )
    
    return fig

def main():
    """Main Streamlit application"""
    st.title("📈 Stock Trading Suggestion System")
    st.markdown("---")
    
    # Sidebar for inputs
    st.sidebar.header("Configuration")
    
    # Stock ticker input
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL", help="Enter stock symbol (e.g., AAPL, TSLA, RELIANCE.NS)")
    
    # Date range selector
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    # Get suggestions button
    get_suggestions = st.sidebar.button("Get Suggestions", type="primary")
    
    # Additional options
    st.sidebar.markdown("---")
    st.sidebar.subheader("Additional Options")
    show_backtest = st.sidebar.checkbox("Show Backtest Results")
    enable_download = st.sidebar.checkbox("Enable CSV Download")
    
    if get_suggestions:
        if ticker:
            with st.spinner(f"Fetching data for {ticker.upper()}..."):
                # Fetch and process data
                data = get_data(ticker.upper(), start_date, end_date)
                
                if data is not None:
                    # Calculate indicators and generate signals
                    data_with_indicators = calculate_indicators(data)
                    data_with_signals = generate_signals(data_with_indicators)
                    
                    if data_with_signals is not None:
                        # Display latest signal
                        st.subheader("📊 Latest Trading Signal")
                        
                        latest_data = data_with_signals.iloc[-1]
                        latest_signal = latest_data['Signal']
                        latest_confidence = latest_data['Confidence']
                        latest_price = latest_data['Close']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            signal_color = {"Buy": "🟢", "Sell": "🔴", "Hold": "🟡"}[latest_signal]
                            st.metric("Signal", f"{signal_color} {latest_signal}")
                        
                        with col2:
                            st.metric("Confidence", f"{latest_confidence:.1f}%")
                        
                        with col3:
                            st.metric("Current Price", f"${latest_price:.2f}")
                        
                        with col4:
                            daily_change = ((latest_price - data_with_signals['Close'].iloc[-2]) / data_with_signals['Close'].iloc[-2]) * 100
                            st.metric("Daily Change", f"{daily_change:.2f}%")
                        
                        # Display technical indicators summary
                        st.subheader("📋 Technical Indicators Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("SMA 20", f"${latest_data['SMA_20']:.2f}")
                        
                        with col2:
                            st.metric("SMA 50", f"${latest_data['SMA_50']:.2f}")
                        
                        with col3:
                            st.metric("RSI", f"{latest_data['RSI']:.1f}")
                        
                        with col4:
                            st.metric("MACD", f"{latest_data['MACD']:.3f}")
                        
                        # Plot charts
                        st.subheader("📈 Interactive Charts")
                        chart = plot_charts(data_with_signals, ticker.upper())
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                        
                        # Backtest results
                        if show_backtest:
                            st.subheader("🔄 Backtest Results")
                            
                            backtest_data = simulate_backtest(data_with_signals)
                            if backtest_data is not None:
                                backtest_chart = create_backtest_chart(backtest_data, ticker.upper())
                                if backtest_chart:
                                    st.plotly_chart(backtest_chart, use_container_width=True)
                                
                                # Display backtest metrics
                                final_return = backtest_data['Returns'].iloc[-1]
                                buy_hold_return = ((backtest_data['Close'].iloc[-1] / backtest_data['Close'].iloc[0]) - 1) * 100
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Strategy Return", f"{final_return:.2f}%")
                                with col2:
                                    st.metric("Buy & Hold Return", f"{buy_hold_return:.2f}%")
                                with col3:
                                    outperformance = final_return - buy_hold_return
                                    st.metric("Outperformance", f"{outperformance:.2f}%")
                        
                        # CSV Download
                        if enable_download:
                            st.subheader("💾 Download Data")
                            
                            # Prepare data for download
                            download_data = data_with_signals[['Open', 'High', 'Low', 'Close', 'Volume', 
                                                             'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 
                                                             'Signal', 'Confidence']].copy()
                            
                            # Convert to CSV
                            csv_buffer = io.StringIO()
                            download_data.to_csv(csv_buffer)
                            csv_data = csv_buffer.getvalue()
                            
                            st.download_button(
                                label="Download Signal Data as CSV",
                                data=csv_data,
                                file_name=f"{ticker.upper()}_trading_signals_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                        
                        # Signal Statistics
                        st.subheader("📊 Signal Statistics")
                        
                        signal_counts = data_with_signals['Signal'].value_counts()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            buy_count = signal_counts.get('Buy', 0)
                            st.metric("Buy Signals", buy_count)
                        
                        with col2:
                            sell_count = signal_counts.get('Sell', 0)
                            st.metric("Sell Signals", sell_count)
                        
                        with col3:
                            hold_count = signal_counts.get('Hold', 0)
                            st.metric("Hold Signals", hold_count)
                        
                        # Display recent signals table
                        st.subheader("📋 Recent Signals")
                        recent_signals = data_with_signals[data_with_signals['Signal'] != 'Hold'].tail(10)
                        
                        if not recent_signals.empty:
                            display_signals = recent_signals[['Close', 'Signal', 'Confidence', 'RSI', 'MACD']].copy()
                            display_signals['Close'] = display_signals['Close'].round(2)
                            display_signals['Confidence'] = display_signals['Confidence'].round(1)
                            display_signals['RSI'] = display_signals['RSI'].round(1)
                            display_signals['MACD'] = display_signals['MACD'].round(3)
                            
                            st.dataframe(display_signals, use_container_width=True)
                        else:
                            st.info("No recent buy/sell signals found in the selected date range.")
                    
                    else:
                        st.error("Failed to process indicators and generate signals.")
                else:
                    st.error(f"Failed to fetch data for {ticker.upper()}. Please check the ticker symbol and try again.")
        else:
            st.warning("Please enter a valid stock ticker.")
    
    else:
        # Display instructions when no data is loaded
        st.info("👈 Enter a stock ticker and click 'Get Suggestions' to start analyzing.")
        
        st.markdown("### How to use this system:")
        st.markdown("""
        1. **Enter a stock ticker** (e.g., AAPL, TSLA, GOOGL, RELIANCE.NS for Indian stocks)
        2. **Select date range** for historical analysis
        3. **Click 'Get Suggestions'** to fetch data and generate signals
        
        ### Signal Logic:
        - **🟢 Buy**: SMA-20 > SMA-50 AND RSI < 70
        - **🔴 Sell**: SMA-20 < SMA-50 AND RSI > 30
        - **🟡 Hold**: All other conditions
        
        ### Features:
        - Real-time technical indicators (SMA, RSI, MACD)
        - Interactive charts with buy/sell signals
        - Signal confidence scoring
        - Optional backtesting simulation
        - CSV data export
        """)

if __name__ == "__main__":
    main()