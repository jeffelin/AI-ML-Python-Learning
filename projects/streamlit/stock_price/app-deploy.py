import yfinance as yf 
import streamlit as st

st.write("""

## Simple Stock Price App 

Shown are the stock closing price and volume of Google!

"""

)

### look at the processes for working closely with this data science app

tickersymbol = 'GOOGL'

tickerdata = yf.Ticker(tickersymbol)

tickerDf = tickerdata.history(period = '1d', start = '2010-5-31', end = '2020-5-31')

st.write("### Closing Price")
st.line_chart(tickerDf.Close)


st.write("### Volume")
st.line_chart(tickerDf.Volume)

# credits --> https://github.com/dataprofessor/streamlit_freecodecamp/tree/main/app_1_simple_stock_price