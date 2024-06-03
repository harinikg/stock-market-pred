import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import altair as alt
from statsmodels.tsa.ar_model import AutoReg
import numpy as np

def get_stock_data(company, start_date, end_date):
    ticker = yf.Ticker(company)
    df = ticker.history(start=start_date, end=end_date)
    return df

def predict_future_prices(df, steps):
    model = AutoReg(df['Close'], lags=1)
    model_fit = model.fit()
    future_dates = [df.index[-1] + datetime.timedelta(days=i) for i in range(1, steps+1)]
    future_prices = model_fit.predict(start=len(df), end=len(df)+steps-1, dynamic=False)

    volatility = future_prices.std() * 0.05
    future_prices_with_noise = future_prices + np.random.normal(0, volatility, len(future_prices))
    return future_dates, future_prices_with_noise

def main():
    st.title('Stock Price Viewer')

    companies = {
        "Google": "GOOGL",
        "Amazon": "AMZN",
        "Apple": "AAPL",
        "Microsoft": "MSFT",
    }

    company = st.selectbox('Select a Company', list(companies.keys()))

    time_periods = {
        "1 Week": 7,
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365
    }

    time_period = st.selectbox('Select Time Period for Historical Data', list(time_periods.keys()))

    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=time_periods[time_period])
    end_date = today

    st.write(f"Displaying data for {company} from {start_date} to {end_date}")

    df = get_stock_data(companies[company], start_date, end_date)

    if df.empty:
        st.error("No data available for the selected company and time period.")
    else:
        st.write("Today's Data:")
        if not df.empty:
            today_data = df.iloc[-1]
            st.write(today_data)

        st.write("Historical Data:")
        st.write(df)


        historical_chart = alt.Chart(df.reset_index()).mark_line().encode(
            x='Date:T',
            y='Close:Q',
            tooltip=['Date', 'Close']
        ).properties(
            width=600,
            height=300
        ).interactive()

        st.altair_chart(historical_chart, use_container_width=True)


        st.write("Future Price Prediction:")
        future_time_period = st.selectbox('Select Time Period for Future Prediction', list(time_periods.keys()))
        future_steps = time_periods[future_time_period]
        future_dates, future_prices = predict_future_prices(df, future_steps)
        future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_prices})
        st.write(future_df)


        min_future_price = min(future_prices)
        max_future_price = max(future_prices)
        min_domain = min_future_price - (min_future_price % 0.25)
        max_domain = max_future_price + (0.25 - (max_future_price % 0.25))

        future_chart = alt.Chart(future_df).mark_line(color='red').encode(
            x='Date:T',
            y=alt.Y('Predicted Price:Q', scale=alt.Scale(domain=[min_domain, max_domain])),
            tooltip=['Date', 'Predicted Price']
        ).properties(
            width=600,
            height=300
        ).interactive()

        st.altair_chart(future_chart, use_container_width=True)


        if st.button('Export Data as CSV'):
            st.write(df.to_csv(index=False))

if __name__ == "__main__":
    main()
