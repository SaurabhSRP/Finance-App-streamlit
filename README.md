# Finance-App-streamlit 
Click on the link to access the app on Streamlit cloud https://finance-app-app-saurabh.streamlit.app/

#Project Overview
- The project showcases a way to create a multipage app for extracting , visualising and predicting stock prices using streamlit while main focus is on "LSTM based stock price prediction".
- Plotly based visualisation method with technical charts is showcased which is commonly encountered and used by many stock traders
- Stacked LSTM model is used for prediction using keras.
- This webpage consists of three main pages and details have been mention below.

***Check out the short video of the app also reduce the playback speed of the video***


https://github.com/SaurabhSRP/Finance-App-streamlit/assets/108528607/be49337c-6c4a-4bcb-9297-194e3dde0f16

# Code and Resources Used
- ***Data API:*** yfinance , Aplha Vantage
- ***Packages:***  streamlit,pandas,numpy,DateTime,yfinance,plotly==5.15.0,requests,scikit-learn,tensorflow,
- ***Webpage*** Streamlit

  ***Install all requirements for the web app using*** pip install -r requirements.txt

# Webpage details and Snapshots.

## Page 1 - Home.py 
- It showcases the company profile based on the stock code. few details such as stock holders , mutual fund holders , balance sheet and also the data based on the timeline user requests.
  ![Alt text](https://github.com/SaurabhSRP/Finance-App-streamlit/blob/main/Project_snapshots/Home1.PNG)
  ![Alt text](https://github.com/SaurabhSRP/Finance-App-streamlit/blob/main/Project_snapshots/Home2.PNG)
  ![Alt text](https://github.com/SaurabhSRP/Finance-App-streamlit/blob/main/Project_snapshots/Home3.PNG)
  ![Alt text](https://github.com/SaurabhSRP/Finance-App-streamlit/blob/main/Project_snapshots/Home4.PNG)

## Page 2 - Technical_charts.py
- Second page in the web app shows all the technical charts which are commanly used by the stock traders with an option to change the input values such as moving averages based on users needs.
- The charts showcased here are Moving Average , Weighted Moving Average , Bollinger Bands , Average True Index in the form of Candle sticks
  ![Alt text](https://github.com/SaurabhSRP/Finance-App-streamlit/blob/main/Project_snapshots/Linechart_snap1.PNG)
  ![Alt text](https://github.com/SaurabhSRP/Finance-App-streamlit/blob/main/Project_snapshots/Candlesticks.PNG)
  ![Alt text](https://github.com/SaurabhSRP/Finance-App-streamlit/blob/main/Project_snapshots/MA.PNG)
  ![Alt text](https://github.com/SaurabhSRP/Finance-App-streamlit/blob/main/Project_snapshots/Weighted%20Moving%20Average.PNG)
  ![Alt text](https://github.com/SaurabhSRP/Finance-App-streamlit/blob/main/Project_snapshots/Bolinger_bands.PNG)
  ![Alt text](https://github.com/SaurabhSRP/Finance-App-streamlit/blob/main/Project_snapshots/AverageTrueIndex.PNG)

## Page 3 - LSTM_Prediction.py
- This webpage helps the user to predict the stock price based on the number of days request to predict using LSTM algorithm
- Every stock price goes through inintal training cycle for 10 epochs before it is ready to predict future prices.
  ![Alt text](https://github.com/SaurabhSRP/Finance-App-streamlit/blob/main/Project_snapshots/LSTM_snap1.PNG)
  ![Alt text](https://github.com/SaurabhSRP/Finance-App-streamlit/blob/main/Project_snapshots/LSTM_snap2.PNG)
  ![Alt text](https://github.com/SaurabhSRP/Finance-App-streamlit/blob/main/Project_snapshots/LSTM_snap3.PNG)
  ![Alt text](https://github.com/SaurabhSRP/Finance-App-streamlit/blob/main/Project_snapshots/LSTM_snap4.PNG)

### HOPE YOU HAD FUN , PLEASE DROP YOUR COMMENTS/SUGGESTIONS 



