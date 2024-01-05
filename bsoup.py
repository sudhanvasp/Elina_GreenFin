# Importing required libraries
import gradio as gr
import yfinance as yf
import matplotlib.pyplot as plt


# Function to fetch and plot company stock trends
def plot_stock_trend(stock_ticker):
    # Fetch historical stock price data for the last 60 days
    company = yf.Ticker(stock_ticker)
    hist_data = company.history(period="60d")

    # Plotting the Closing Price trend
    plt.figure(figsize=(10, 6))
    plt.plot(hist_data['Close'])
    plt.title(f'{stock_ticker} Stock Price Trend - Last 60 Days')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.grid(True)
    plt.xticks(rotation=45)

    # Return the plot
    return plt


# Creating Gradio interface
interface = gr.Interface(
    fn=plot_stock_trend,
    inputs=gr.Textbox(label="Enter Stock Ticker (e.g., AAPL for Apple)"),
    outputs=gr.Plot()
)

# Launch the interface
interface.launch()