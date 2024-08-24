
# Stock Dashboard

## Overview

The Stock Dashboard is a web application built with Dash that provides real-time stock market data and analysis. It features interactive candlestick charts with RSI (Relative Strength Index) overlays, and predictive analysis of stock prices. The application is designed with a modern dark mode theme for better readability and user experience.

## Features

- **Real-time Data**: Fetches stock data at one-minute intervals.
- **Interactive Charts**: Displays candlestick charts with RSI and predicted future prices.
- **Modern Design**: Dark mode styling for a sleek look.
- **Predictive Analysis**: Provides predictions for stock prices based on historical data using linear regression.

## Technologies

- **Dash**: A web application framework for Python.
- **Plotly**: For interactive graphing.
- **yfinance**: To fetch historical stock data.
- **pandas**: For data manipulation.
- **numpy**: For numerical operations.
- **scikit-learn**: For machine learning models.
- **Flask-Caching**: For caching data to improve performance.

## Installation

To set up the Stock Dashboard on your local machine, follow these steps:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/desmet-lars/Stock-Analyst.git
   cd Stock-Analyst
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Application**

   ```bash
   python app.py
   ```

   The application will start, and you can view it by navigating to `http://127.0.0.1:8050/` in your web browser.

## Usage

- The dashboard updates every minute to reflect the latest stock data.
- Use the candlestick charts to analyze historical price movements.
- The RSI line indicates potential overbought or oversold conditions.
- The predictive analysis section provides future price predictions and recommendations based on recent trends.

## Configuration

- **POPULAR_STOCKS**: List of stock symbols to display on the dashboard. You can modify this list to include other stocks.
- **CACHE_DIR**: Directory where cache files are stored. Adjust the path if needed.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push to the feature branch.
5. Create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Dash** and **Plotly** for providing the tools to build interactive web applications.
- **yfinance** for easy access to financial data.

## Contact

For any questions or feedback, please contact [your-email@example.com](mailto:your-email@example.com).

### Notes:
1. **Repository URL**: Replace `https://github.com/yourusername/stock-dashboard.git` with the actual URL of your GitHub repository.
2. **Email Address**: Replace `your-email@example.com` with your contact email.
3. **License**: Ensure you include or link to a `LICENSE` file if your project uses a specific license.

Feel free to adjust the content to better fit your project's specifics or any additional information you want to provide!
