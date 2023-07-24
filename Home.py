import streamlit as st
import pandas as pd
import numpy as np
import datetime 
from datetime import date, timedelta
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import requests

st.set_page_config(layout='wide',initial_sidebar_state='expanded',page_title="Home",page_icon=":house:")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)


#-------------------------TITLE------------------------------------------------------------#
st.title("""Greetings from :chart_with_upwards_trend: STOCK-NOTCH """)
st.markdown(''' --- ''')

#----------------------------SIDEBAR---------------------------------------------------------#
st.sidebar.success('Select a Page Above')
st.sidebar.title(":chart_with_upwards_trend: STOCK-NOTCH")

st.sidebar.header("One stop destination for all data and prediction needs")

st.sidebar.subheader('STOCK CODE')
stock_title = st.sidebar.text_input('STOCK CODE',placeholder="*ex Microsoft = MSFT ",label_visibility="collapsed")

st.sidebar.subheader('Start Date')
start_date = st.sidebar.text_input('Start Date',placeholder="YYYY-MM-DD ",label_visibility="collapsed")

st.sidebar.subheader('End Date')
end_date = st.sidebar.text_input('End Date',placeholder="YYYY-MM-DD ",label_visibility="collapsed")



#st.sidebar.subheader('Number of Days stock history?')
#number_of_days=st.sidebar.number_input('Number of days old data',label_visibility="collapsed")
#st.sidebar.write("(Min. 365 days)")


#---------------------------------------------------stock data-----------------------------------#
def stock_data(stock_title,start_date,end_date):
    #today=date.today()
    #end_date=today.strftime("%Y-%m-%d")
    #start_date=date.today()-timedelta(days=number_of_days)
    #start_date=start_date.strftime("%Y-%m-%d") 
    data=yf.download(stock_title,start=start_date,end=end_date,progress=False)
    data["Date"]=data.index
    data=data[["Date", "Open", "High", "Low", 
             "Close", "Adj Close", "Volume"]]
    data.reset_index(drop=True,inplace=True)
    return data

#---------------------------------------------------stock financial statements-----------------------------------#
def stock_statements(stock_title):
    stock=yf.Ticker(stock_title)
    income=stock.income_stmt
    balance=stock.balance_sheet
    cash=stock.cashflow
    institute=stock.institutional_holders
    mutualf=stock.mutualfund_holders
    return income,balance,cash,institute,mutualf

#---------------------------------------------------stock company overview-----------------------------------#
def stock_info(stock_title):
    url = 'https://www.alphavantage.co/query?function=OVERVIEW&symbol={}&apikey=5MPQWMV39FQ2FWBQ'.format(stock_title)
    r = requests.get(url)
    info = r.json()
    return info


#---------------------------------------------------submit button-----------------------------------#
if st.sidebar.button('Submit'):
    info=stock_info(stock_title)
    st.title(info['Name'])
    st.write(info['Description'])

    col1,col2=st.columns(2)
    with col1:
        st.header("Sector")
        st.subheader(info['Sector'])

    with col2:
        st.header("Industry")
        st.subheader(info['Industry'])
    
    col3,col4,col5,col6=st.columns(4)
    with col3:
        st.header("P/E Ratio")
        st.subheader(info['PERatio'])

    with col4:
        st.header("52 Week High")
        st.subheader(info['52WeekHigh'])
    
    with col5:
        st.header("52 Week low")
        st.subheader(info['52WeekLow'])
    
    with col6:
        st.header("Market-Cap")
        st.subheader(info['MarketCapitalization'])
    
     #---------------------------------------tab font size and tabs-------------------------------------#
    font_css = """
    <style>
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
        font-size: 20px;
        }
        </style>
        """
    
    st.write(font_css, unsafe_allow_html=True)

    tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(["Data","Income Statement","Balance Sheets","Cashflow","Institutional Holders","mutualfund Institute Holders"])

    income,balance,cash,institute,mutualf = stock_statements(stock_title)

    with tab1:
        st.header("Data")
        st.dataframe(stock_data(stock_title,start_date,end_date),use_container_width=True)
    
    with tab2:
        st.header("Income statement")
        st.dataframe(income,use_container_width=True)

    with tab3:
        st.header("Balance Sheets")
        st.dataframe(balance,use_container_width=True)
    
    with tab4:
        st.header("Cashflow")
        st.dataframe(cash,use_container_width=True)
    
    with tab5:
        st.header("Top 10 Insititutional Holders")
        fig = px.bar(institute, x="Shares", y="Holder", orientation='h',text_auto=True)
        st.plotly_chart(fig,use_container_width=True)


    with tab6:
        st.header("Top 10 Mutual Fund Institute Holders")
        fig = px.bar(mutualf, x="Shares", y="Holder", orientation='h',text_auto=True)
        st.plotly_chart(fig,use_container_width=True)



st.sidebar.markdown(""" Created with :heart: by Saurabh """)

    

   






