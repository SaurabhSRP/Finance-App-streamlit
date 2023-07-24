import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
scaler=MinMaxScaler(feature_range=(0,1))
from datetime import date,datetime,timedelta

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

st.set_page_config(layout='wide',initial_sidebar_state='expanded',page_title="LSTM_prediction",page_icon=":bar_chart:")

st.title("Stock Price Prediction using LSTM")
st.markdown(''' --- ''')
st.header(""" We are going to predict only the "Close" price of the stock """)


st.sidebar.success('Please Input Data again for Stock Prediction')
st.sidebar.subheader('STOCK CODE')
stock_title = st.sidebar.text_input('STOCK CODE',placeholder="*ex Microsoft = MSFT ",label_visibility="collapsed")

st.sidebar.subheader('Start Date')
start_date = st.sidebar.text_input('Start Date',placeholder="YYYY-MM-DD ",label_visibility="collapsed")
st.sidebar.write(' LSTMs are data hungry please provide data more than 365 days')

st.sidebar.subheader('End Date')
end_date = st.sidebar.text_input('End Date',placeholder="YYYY-MM-DD ",label_visibility="collapsed")

st.sidebar.markdown(""" Created with :heart: by Saurabh """)




def stock_data(stock_title,start_date,end_date):
    data=yf.download(stock_title,start=start_date,end=end_date,progress=False)
    data["Date"]=data.index
    data=data[["Date", "Open", "High", "Low", 
             "Close", "Adj Close", "Volume"]]
    data.reset_index(drop=True,inplace=True)
    return data

st.subheader("""visualise the Closing price of {} """.format(stock_title))

try:
    data=stock_data(stock_title,start_date,end_date)
except:
    st.write("waiting for input")

try:
    plot1=px.line(data['Close'])
    plot1.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(plot1,use_container_width=True)
except:
    st.write("waiting for input")

def create_dataset(dataset, time_step):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

try:
    st.header("Lets prepare the LSTM model and visualise the steps as we train the model")
    #####Consider the close price in a dataset#######
    df=data['Close']
    ##### Scale the dataset in the range of 0-1
    df=scaler.fit_transform(np.array(df).reshape(-1,1))
    #####Consider a training size and split the data#########
    training_size=int(len(df)*0.6)
    st.write(""" - We are considering 60 percent of the data for training,in this case {} is the training size """.format(training_size))
    test_size=len(df)-training_size
    train_data,test_data=df[0:training_size,:],df[training_size:len(df),:1]
    st.write("""- creating the dataset with time step of 100 or in other terms with 100 features. which is required for time series based data.""")
    ####CREATE X_TRAIN AND Y_TRAIN BASED ON TIMESTEP#########
    time_step=100
    X_train,y_train=create_dataset(train_data,time_step)
    X_test,y_test=create_dataset(test_data,time_step)
    st.write(""" - X_train shape  {}  """.format(X_train.shape))
    ####MODEL RESHAPE for LSTM input #######
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    ####CREATE LSTM MODEL ########
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(100,1))) #input_shape should be time_steps,features
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    st.write("- LSTM model summary for reference")
    model.summary(print_fn=lambda x:st.text(x))
    st.error("Training the Model for 10 epochs, wait for success message")
    ###Train the model
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64,verbose=1)
    st.success("Training Compeleted, Creating test dataset. Input number of days to predict when input window pops up")
    #predict the model
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    #inverse transform the scaled model
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
    #RMSE of the model
    rmse=math.sqrt(mean_squared_error(y_train,train_predict))
    st.write(""" -The RMSE value gives the indication of accuracy of our model , lower the value better the prediction. In this case RMSE value is {} """.format(rmse))
    #####Creating the future prediction dataset#############
except:
    st.write("waiting for input")


st.subheader('No. of Days to Predict?')
pred_days=st.number_input('pred_days',label_visibility='collapsed')
if st.button('Predict'):
    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    lst_output=[]
    n_steps=100
    i=0
    while(i< int(pred_days)):
         if (len(temp_input)>100):
              x_input=np.array(temp_input[1:])
              x_input=x_input.reshape(1,-1)
              x_input=x_input.reshape(1,-1)
              x_input=x_input.reshape((1,n_steps,1))
              ypred=model.predict(x_input,verbose=0)
              temp_input.extend(ypred[0].tolist())
              temp_input=temp_input[1:]
              lst_output.extend(ypred.tolist())
              i=i+1
         else:
              x_input=x_input.reshape(1,n_steps,1)
              ypred=model.predict(x_input,verbose=0)
              temp_input.extend(ypred[0].tolist())
              lst_output.extend(ypred.tolist())
              i=i+1
    lst_output=scaler.inverse_transform(lst_output)
    lst_output.reshape(1,-1)
    lst_output=np.squeeze(lst_output)
    list=lst_output.tolist()
    end=end_date
    end=datetime.strptime(end,"%Y-%m-%d")
    dates_list=[]
    for x in range(0,int(pred_days)):
         dates=end + timedelta(days=x)
         dates=dates.strftime("%Y-%m-%d")
         dates_list.append(dates)
    pred_plot={"Date":dates_list,"Close":list}
    pred_df=pd.DataFrame(pred_plot)
    pred_df["Date"]=pd.to_datetime(pred_df["Date"])
    plot_data=data[["Date","Close"]]
    frames=[plot_data,pred_df]
    result=pd.concat(frames)
    st.dataframe(result)
    train=result[:training_size]
    valid=result[training_size:]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=train["Date"],y=train["Close"],name="Train_data",line=dict(color="#339FFF")))
    fig.add_trace(go.Scatter(x=valid["Date"],y=valid["Close"],name="Test_data",line=dict(color="#33FF3C")))
    fig.add_trace(go.Scatter(x=pred_df["Date"],y=pred_df["Close"],name="Prediction",line=dict(color="#0000FF")))
    fig.update_layout( xaxis_rangeslider_visible=False)
    st.plotly_chart(fig,use_container_width=True)

