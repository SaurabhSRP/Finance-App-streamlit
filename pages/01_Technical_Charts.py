import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


st.set_page_config(layout='wide',initial_sidebar_state='expanded',page_title="Technical_chart",page_icon=":bar_chart:")

st.title("Technical Charts")
st.markdown(''' --- ''')

st.sidebar.success('Please Input Data again for Technical charts')
st.sidebar.subheader('STOCK CODE')
stock_title = st.sidebar.text_input('STOCK CODE',placeholder="*ex Microsoft = MSFT ",label_visibility="collapsed")

st.sidebar.subheader('Start Date')
start_date = st.sidebar.text_input('Start Date',placeholder="YYYY-MM-DD ",label_visibility="collapsed")

st.sidebar.subheader('End Date')
end_date = st.sidebar.text_input('End Date',placeholder="YYYY-MM-DD ",label_visibility="collapsed")

st.sidebar.markdown(""" Created with :heart: by Saurabh """)



def stock_data(stock_title,start_date,end_date): 
    data=yf.download(stock_title,start=start_date,end=end_date,progress=False)
    data["Date"]=data.index
    data=data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    data.reset_index(drop=True,inplace=True)
    return data

try:
    data=stock_data(stock_title,start_date,end_date)
except:
    st.write("waiting for input")

st.header('Line Charts for indiviual feature')
try:
    #data=stock_data(stock_title,start_date,end_date)
    tab1,tab2=st.tabs(['Open','Close'])
    with tab1:
        plot1=px.line(data,x="Date",y="Open")
        plot1.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(plot1,use_container_width=True)
    
    with tab2:
        plot2=px.line(data,x="Date",y="Close")
        plot2.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(plot2,use_container_width=True) 

    st.header('CandleSticks')
    plot3=go.Figure(data=[go.Candlestick(x=data["Date"],open=data["Open"],high=data["High"],low=data["Low"],close=data["Close"])])
    plot3.update_layout( xaxis_rangeslider_visible=False)
    st.plotly_chart(plot3,use_container_width=True)
except:
    st.write("waiting for input")

def SMA(data,numdays):
  SMA=pd.Series(data['Close'].rolling(numdays).mean(),name='SMA')
  data=data.join(SMA)
  return data

st.header("Three Moving Average (30days-60days-90days)")
col1,col2,col3=st.columns(3)
with col1:
    input1=st.number_input("First MA")
with col2:
    input2=st.number_input("Second MA")
with col3:
    input3=st.number_input("Third MA")


try:
    #data=stock_data(stock_title,start_date,end_date)
    input1=int(input1)
    parameter1=SMA(data,input1)
    parameter1=parameter1=SMA(data,input1).dropna()
    parameter1_sma=parameter1['SMA']
    
    input2=int(input2)
    parameter2=SMA(data,input2)
    parameter2=parameter2.dropna()
    parameter2_sma=parameter2['SMA']

    input3=int(input3)
    parameter3=SMA(data,input3)
    parameter3=parameter3.dropna()
    parameter3_sma=parameter3['SMA']


    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=data["Date"],open=data["Open"],high=data["High"],low=data["Low"],close=data["Close"]))
    fig.add_trace(go.Scatter(x=data["Date"],y=parameter1_sma,line=dict(color="#0000FF"),name="First Input MA"))
    fig.add_trace(go.Scatter(x=data["Date"],y=parameter2_sma,line=dict(color="#800080"),name="Second Input MA"))
    fig.add_trace(go.Scatter(x=data["Date"],y=parameter3_sma,line=dict(color="#ffff00"),name="Third Input MA"))
    fig.update_layout( xaxis_rangeslider_visible=False)
    st.plotly_chart(fig,use_container_width=True)
except:
    st.write("waiting for input")
     

def EWMA(data,EWMA_input):
    EWMA = pd.Series(data['Close'].ewm(span = EWMA_input, min_periods = EWMA_input - 1).mean(), name = 'EWMA') 
    data = data.join(EWMA) 
    return data

st.header("Exponential Weighted Moving Average")
EWMA_input=st.number_input("Input the number of days?")

try:
    #data=stock_data(stock_title,start_date,end_date)
    EWMA=EWMA(data,EWMA_input)
    EWMA = EWMA.dropna()
    EWMA = EWMA['EWMA']
    fig1=go.Figure()
    fig1.add_trace(go.Candlestick(x=data["Date"],open=data["Open"],high=data["High"],low=data["Low"],close=data["Close"]))
    fig1.add_trace(go.Scatter(x=data["Date"],y=EWMA,line=dict(color="#0000FF"),name=" EWMA"))
    fig1.update_layout( xaxis_rangeslider_visible=False)
    st.plotly_chart(fig1,use_container_width=True)
except:
    st.write("waiting for input")



def BBANDS(data,bbands_input):
    MA = data.Close.rolling(window=bbands_input).mean()
    SD = data.Close.rolling(window=bbands_input).std()
    data['MiddleBand'] = MA
    data['UpperBand'] = MA + (2 * SD) 
    data['LowerBand'] = MA - (2 * SD)
    return data


st.header("Bollinger Bands")
bbands_input=st.number_input("Input the number of days for Bollinger Bands moving average?")


try:
    #data=stock_data(stock_title,start_date,end_date)
    bbands_input=int(bbands_input)
    BBANDS=BBANDS(data,bbands_input)
    fig2=go.Figure()
    fig2.add_trace(go.Candlestick(x=BBANDS["Date"],open=BBANDS["Open"],high=BBANDS["High"],low=BBANDS["Low"],close=BBANDS["Close"]))
    fig2.add_trace(go.Scatter(x=BBANDS["Date"],y=BBANDS["UpperBand"],line=dict(color="#89CFF0"),name="UpperBand"))
    fig2.add_trace(go.Scatter(x=BBANDS["Date"],y=BBANDS["MiddleBand"],line=dict(color="#008080"),name="MiddleBand"))
    fig2.add_trace(go.Scatter(x=BBANDS["Date"],y=BBANDS["LowerBand"],line=dict(color="#89CFF0"),name="LowerBand"))
    fig2.update_layout( xaxis_rangeslider_visible=False)
    st.plotly_chart(fig2,use_container_width=True)
except:
    st.write("waiting for input")


def atr(high,low,close,n=14):
    atr=tr = np.amax(np.vstack(((high - low).to_numpy(), (abs(high - close)).to_numpy(), (abs(low - close)).to_numpy())).T, axis=1)
    return pd.Series(tr).rolling(n).mean().to_numpy()

st.header("Average True Index")

try:
    data['ATR']=atr(data['High'],data['Low'],data['Close'],14)
    fig3 =make_subplots(rows=2, cols=1)
    fig3.add_trace(go.Candlestick(x=data["Date"],open=data["Open"],high=data["High"],low=data["Low"],close=data["Close"]),row=1,col=1)
    fig3.add_trace(go.Scatter(x=data["Date"],y=data["ATR"]),row=2,col=1)
    fig3.update_layout( xaxis_rangeslider_visible=False)
    st.plotly_chart(fig3,use_container_width=True)
except:
    st.write("waiting for input")