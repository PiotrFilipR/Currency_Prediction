"""
What the project below wants to show, is that the exchange rates of three currencies - EUR, USD and GBP
are indirectly linked and the current rate of each of them will interact with the rates of the other two.
The following project will try to prove the thesis by conducting an analysis in which, after teaching an algorithm with
dataset based on historical data from the selected period, it will try to predict, based on USD and GBP exchange rates,
EUR exchange rate with an accuracy below the standard deviation.

"""



from forex_python.converter import CurrencyRates
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import datetime as dt
import sys
from PyQt5.QtWidgets import QApplication
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

import tkinter as tk

root = tk.Tk()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

app = QApplication(sys.argv)
screen = app.screens()[0]
my_dpi = screen.physicalDotsPerInch()


## Creating dataset

# dates = []
# USD_values = []
# EUR_values = []
# GBP_values = []
# c = CurrencyRates()
# for year in range(2000,2021):
#     for month in range(1,13):
#         temp_date = dt.datetime(year, month, 1, 0, 0, 0, 0)
#         dates.append(str(year)+ "." +str(month).zfill(2))
#         USD_values.append(c.get_rate('USD', 'PLN', temp_date))
#         EUR_values.append(c.get_rate('EUR', 'PLN', temp_date))
#         GBP_values.append(c.get_rate('GBP', 'PLN', temp_date))

# full_dict = {'date': dates, 'EUR': EUR_values, 'USD': USD_values, 'GBP': GBP_values}
# df = pd.DataFrame(full_dict)
# df.to_csv('waluty.csv', index=False)


currencies = pd.read_csv("waluty.csv")

currencies = currencies[["date", "EUR", "USD", "GBP"]]

print(currencies.shape)


# Test 1

train_1 = currencies[currencies["date"] < 2012.01].copy()
test_1 = currencies[currencies["date"] >= 2012.01].copy()

print(train_1.shape)
print(test_1.shape)

reg = LinearRegression()

predictors = ["USD", "GBP"]
reg.fit(train_1[predictors], train_1["EUR"])

LinearRegression()

predictions = reg.predict(test_1[predictors])

test_1["predictions"] = predictions

# Test 2

train_2 = currencies[currencies["date"] < 2018.01].copy()
test_2 = currencies[currencies["date"] >= 2018.01].copy()

print(train_2.shape)
print(test_2.shape)

reg = LinearRegression()

predictors = ["USD", "GBP"]
reg.fit(train_2[predictors], train_2["EUR"])

LinearRegression()

predictions = reg.predict(test_2[predictors])

test_2["predictions"] = predictions

# Results

print(test_1)

error_1 = mean_absolute_error(test_1["EUR"], test_1["predictions"])

print(error_1)

print(currencies.describe()["EUR"])


print(test_2)

error_2 = mean_absolute_error(test_2["EUR"], test_2["predictions"])

print(error_2)

print(currencies.describe()["EUR"])

## Data visualization

df = pd.read_csv('waluty.csv')
list_of_nans = []
for i in list(train_1["EUR"]):
    list_of_nans.append(np.nan)

df["test_1"] = pd.Series(list(train_1["EUR"]) + list(test_1["predictions"]) )
df["test_1_prediction"] = list_of_nans + list(test_1["predictions"])

list_of_nans = []
for i in list(train_2["EUR"]):
    list_of_nans.append(np.nan)

df["test_2"] = pd.Series(list(train_2["EUR"])+ list(test_2["predictions"]))
df["test_2_prediction"] = list_of_nans + list(test_2["predictions"])

x = [dt.datetime.strptime(str(format(d, '.2f')),'%Y.%m').date() for d in df["date"]]


fig, axs = plt.subplots(3)
orange_patch = mpatches.Patch(color='orange', label='EUR')
red_patch = mpatches.Patch(color='red', label='EUR prediction')
blue_patch = mpatches.Patch(color='blue', label='GBP')
green_patch = mpatches.Patch(color='green', label='USD')
my_yticks = np.linspace(0,7.5,15)
my_xticks = []
for idx, date in enumerate(x):
    if idx%12 == 0:
        my_xticks.append(date)


plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m'))
plt.yticks(my_yticks)
for ax in [axs[0], axs[1], axs[2]]:

    ax.set_ylabel("PLN")
    ax.set_yticks(my_yticks)

    if ax == axs[0]:
        ax.set_title("Real EUR rate", fontsize = 15)
        ax.plot(x, df["EUR"], label = "EUR", color='orange')
        ax.legend(handles=[orange_patch, red_patch, blue_patch, green_patch],bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0)
    elif ax == axs[1]:
        ax.set_title("Prediction from 2012", fontsize = 11)
        ax.plot(x, df["test_1"], label = "EUR", color='orange')
        ax.plot(x, df["test_1_prediction"], label = "EUR", color='red')
    elif ax == axs[2]:
        ax.set_title("Prediction from 2018", fontsize = 11)
        ax.plot(x, df["test_2"], label = "EUR", color='orange')
        ax.plot(x, df["test_2_prediction"], label = "EUR", color='red')
    ax.plot(x, df["GBP"], label = "GBP", color='blue')
    ax.plot(x, df["USD"], label = "USD", color='green')


# fig.figsize=(screen_width/my_dpi*0.9, screen_height/my_dpi*0.8))
fig.set_figwidth(screen_width/my_dpi*0.9)
fig.set_figheight(screen_height/my_dpi*0.8)
plt.show()

# Conclusions

"""

Analyzing the two tests carried out, we can draw the following conclusions:

- Our initial thesis turned out to be true - analyzing the exchange rates of three currencies, 
we can conclude that the current exchange rate of these currencies is indirectly linked and based on the exchange rate 
of two of the three currencies, we are able to find the appropriate one error in estimating the value of the third 
course,

- Two tests that used different sizes of data to train the model showed that the larger the sample of data to train is,
the smaller the final error we achieve, and thus the more accurate our predictions are.


"""