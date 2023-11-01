import pandas as pd 
import datetime
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


#loading the CSV file containinge data

Inndata = pd.read_csv("TSLA.csv")

Inndata["Date"] = pd.to_datetime(Inndata["Date"])

#Sorting data based on date, and adjusting "Date" as an index 

Inndata = Inndata.sort_values(by="Date")
Inndata.set_index("Date", inplace= True)


X = Inndata.index.to_julian_date().values.reshape(-1,1)
y = Inndata["Close"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

modell = LinearRegression()
modell.fit(X_train, y_train)

y_predic = modell.predict(X_test)

# r_squared : represent goodness of fit to a regression model.Value between 0 and 1 where 1 is perfect fit.
r_squared = r2_score(y_test, y_predic)
# MeanAE : represents mean absolute error betweeen paired observations.
MeanAE = mean_absolute_error(y_test, y_predic)



print(f"R squared value Score: {r_squared} ")
print(f"Mean absolute error: {MeanAE}")




