import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import streamlit as st
Data=pd.read_csv(filepath_or_buffer="./Advertising Budget and Sales.csv",index_col=[0])
# ad.head()
columns=["tv","radio","news","sales"]
Data.columns=columns

dict_scaler=dict()

for col in columns:
    dict_scaler[col]=StandardScaler().fit(Data[[col]])

df=Data.copy()

# radio_scaler=StandardScaler().fit(X=Data[["radio"]])
# radio_scaler
# tv_scaler=StandardScaler().fit(X=Data[["tv"]])
# tv_scaler
# news_scaler=StandardScaler().fit(X=Data[["news"]])
# news_scaler
# sales_scaler=StandardScaler().fit(X=Data[["sales"]])
# sales_scaler

for col in columns:
    df[col]=dict_scaler[col].transform(X=Data[[col]])

# Data1=Data.copy()
# Data1["tv"]=tv_scaler.transform(X=Data[["tv"]])
# Data1["radio"]=radio_scaler.transform(X=Data[["radio"]])
# Data1["news"]=news_scaler.transform(X=Data[["news"]])
# Data1["sales"]=sales_scaler.transform(X=Data[["sales"]])
# df.head()

target="sales"
features=df.columns.to_list()
features.remove(target)
X=df[features].values
Y=df[target].values
# print(X[0:5, :])
# print(Y[0:5,])
robot =LinearRegression().fit(X=X,y=Y)
st.title(body="Predict Sales By Advertisement Type")
# robot
user_input =dict()
for col in features:
    user_input[col]=st.number_input(label=col,value=Data[col].mean())
df_input=pd.DataFrame(data=[user_input])

df_input=pd.DataFrame(data=[user_input],columns=features)
df_scaled=df_input.copy()
for col in features:
    df_scaled[col]=dict_scaler[col].transform(X=df_input[[col]])
X_test=df_scaled.values
st.dataframe(data=df_input)
st.dataframe(data=df_scaled)
y_pred=robot.predict(X=X_test)
y_pred=dict_scaler[target].inverse_transform(y_pred.reshape(-1, 1))
st.write(y_pred)

import matplotlib.pyplot as plt
import seaborn as sns 

fig, ax=plt.subplots()
ax.hist(x=Data[target])
st.pyplot(fig)

fig,ax=plt.subplots()
sns.heatmap(data=Data.corr(),ax=ax)
st.pyplot(fig)

