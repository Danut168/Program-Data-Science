import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import streamlit as st

data=pd.read_csv('C:/Users/vanna/OneDrive - Royal School of Administration/Desktop/Program Data Science/Program_Car_Sale.csv')
# data.head(10)

### Filtering the rows that has a value in the column - Sales_in_thousands

modified_dataset = data[data['Sales_in_thousands'].notna()]
# modified_dataset
#Fill missing 
### Replacing the missing values in the column - __year_resale_value using median

year_index = list(~modified_dataset['four_year_resale_value'].isnull())
median_year = np.median(modified_dataset['four_year_resale_value'].loc[year_index])
# median_year

### Replacing the missing values of the column - __year_resale_value in the dataset

modified_dataset['four_year_resale_value'].fillna(median_year, inplace = True)

### Replacing the missing values in the column - __year_resale_value using median

year_index = list(~modified_dataset['Fuel_efficiency'].isnull())
median_year = np.median(modified_dataset['Fuel_efficiency'].loc[year_index])
# median_year

### Replacing the missing values of the column - __year_resale_value in the dataset

modified_dataset['Fuel_efficiency'].fillna(median_year, inplace = True)

### Replacing the missing values in the column - __year_resale_value using median

year_index = list(~modified_dataset['Price_in_thousands'].isnull())
median_year = np.median(modified_dataset['Price_in_thousands'].loc[year_index])
# median_year

### Replacing the missing values of the column - __year_resale_value in the dataset

modified_dataset['Price_in_thousands'].fillna(median_year, inplace = True)

### Replacing the missing values in the column - __year_resale_value using median

year_index = list(~modified_dataset['Curb_weight'].isnull())
median_year = np.median(modified_dataset['Curb_weight'].loc[year_index])
# median_year

### Replacing the missing values of the column - __year_resale_value in the dataset

modified_dataset['Curb_weight'].fillna(median_year, inplace = True)

### Replacing the missing values in the column - __year_resale_value using median

year_index = list(~modified_dataset['Power_perf_factor'].isnull())
median_year = np.median(modified_dataset['Power_perf_factor'].loc[year_index])
# median_year

### Replacing the missing values of the column - __year_resale_value in the dataset

modified_dataset['Power_perf_factor'].fillna(median_year, inplace = True)

columns=["sale","fouryear","price","enginesize","horsepower","wheelbase","width","length","curbweight","fuelcapital","fuelefficiency","powerfactor"]
modified_dataset.columns=columns
# modified_dataset.head()

dict_scaler=dict()

for col in columns:
    dict_scaler[col]=StandardScaler().fit(modified_dataset[[col]])

df=modified_dataset.copy()
for col in columns:
    df[col]=dict_scaler[col].transform(X=modified_dataset[[col]])

from sklearn.linear_model import LinearRegression
target="sale"
features=df.columns.to_list()
features.remove(target)
X=df[features].values
Y=df[target].values
# print(X[0:5, :])
# print(Y[0:5,])
robot =LinearRegression().fit(X=X,y=Y)
st.title(body="Car Sale Web App", anchor=False)
st.title(body="Predict Car Sale")

user_input =dict()
for col in features:
    user_input[col]=st.sidebar.number_input(label=col,value=modified_dataset[col].mean())
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
ax.hist(x=modified_dataset[target])
st.pyplot(fig)

fig,ax=plt.subplots()
sns.heatmap(data=modified_dataset.corr(),ax=ax)
st.pyplot(fig)


