# Ex.No: 08 MOVING AVERAGE MODEL AND EXPONENTIAL SMOOTHING

## AIM:
  To implement Moving Average Model and Exponential smoothing using Astrobiological dataset.

## ALGORITHM:
 1. Import necessary libraries
 2. Read the temperature time series data from a CSV file, Display the shape and the first 20 rows of the dataset
 3. Set the figure size for plots
 4. Suppress warnings
 5. Plot the first 50 values of the 'Value' column
 6. Perform rolling average transformation with a window size of 5
 7. Display the first 10 values of the rolling mean
 8. Perform rolling average transformation with a window size of 10
 9. Create a new figure for plotting, Plot the original data and fitted value
 10. Show the plot
 11. Also perform exponential smoothing and plot the graph

## PROGRAM: 
```
Developed by: Naveenkumar M
Reg No: 212224230182

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load and inspect the dataset
df = pd.read_csv('/content/AirPassengers (1).csv')
print("Column names:", df.columns.tolist())
print(df.head())

# Rename columns to standard names if needed
df.columns = ['Date', 'Passengers']  # Adjust as per actual column names

# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set date as index
df.set_index('Date', inplace=True)

# Ensure 'Passengers' column is numeric
df['Passengers'] = pd.to_numeric(df['Passengers'], errors='coerce')

# Drop NaN values
df.dropna(inplace=True)

# Plot original time series
plt.figure(figsize=(10, 6))
plt.plot(df['Passengers'], label='Original Data', marker='o')
plt.title('Monthly Airline Passengers')
plt.ylabel('Number of Passengers')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.show()

# Rolling averages
rolling_mean_5 = df['Passengers'].rolling(window=5).mean()
rolling_mean_10 = df['Passengers'].rolling(window=10).mean()

plt.figure(figsize=(10, 6))
plt.plot(df['Passengers'], label='Original Data', marker='o')
plt.plot(rolling_mean_5, label='Rolling Mean (5)', linestyle='--')
plt.plot(rolling_mean_10, label='Rolling Mean (10)', linestyle='--')
plt.title('Rolling Averages')
plt.legend()
plt.grid(True)
plt.show()

# Exponential Smoothing
exp_smoothing = SimpleExpSmoothing(df['Passengers']).fit(smoothing_level=0.2, optimized=False)
exp_smoothed = exp_smoothing.fittedvalues

plt.figure(figsize=(10, 6))
plt.plot(df['Passengers'], label='Original Data')
plt.plot(exp_smoothed, label='Exponential Smoothing', linestyle='--')
plt.title('Exponential Smoothing')
plt.legend()
plt.grid(True)
plt.show()

# ACF and PACF plots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_acf(df['Passengers'], lags=20, ax=plt.gca())
plt.title('Autocorrelation (ACF)')

plt.subplot(1, 2, 2)
plot_pacf(df['Passengers'], lags=20, ax=plt.gca())
plt.title('Partial Autocorrelation (PACF)')
plt.tight_layout()
plt.show()

# Forecast next 3 months
forecast = exp_smoothing.forecast(steps=3)
future_dates = pd.date_range(start=df.index[-1], periods=4, freq='MS')[1:]

plt.figure(figsize=(10, 6))
plt.plot(df['Passengers'], label='Original Data')
plt.plot(future_dates, forecast, label='Forecast (Next 3)', marker='x')
plt.title('Forecast of Airline Passengers')
plt.legend()
plt.grid(True)
plt.show()
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/5dae8dc4-6dab-4dc4-8d9c-4cc95108fd04)
![image](https://github.com/user-attachments/assets/d7b6abf7-20ab-40db-8ced-b5937b41624c)
![image](https://github.com/user-attachments/assets/8e9330c6-6ec5-4630-abf0-282c18a1dfbb)
![image](https://github.com/user-attachments/assets/6bf3f16a-d231-4e1b-ab1e-c1be31b368f9)
![image](https://github.com/user-attachments/assets/8d839485-3550-46c9-97fd-7fa90b3f4353)

## RESULT:
Thus the python code successfully implemented for the Moving Average Model and Exponential smoothing for daily minimum temperature dataset.
