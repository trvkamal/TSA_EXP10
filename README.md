# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### NAME : kamalesh v
### REG NO : 212222240042


### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
Import necessary library:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
```
Load and clean data:
```
data = pd.read_csv('GoogleStockPrices.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data_ts = data['Close']
```
Plot GDP Trend:
```
plt.figure(figsize=(12, 6))
plt.plot(data_ts)
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.title('Google Stock Price Time Series (Close)')
plt.grid(True)
plt.show()
```
Check Stationarity :
```
def check_stationarity(timeseries):
    # Perform Dickey-Fuller test
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

print("\n--- Stationarity Check on Closing Prices ---")
check_stationarity(data_ts)
```
Plot ACF and PCF:
```
plt.figure(figsize=(12, 5))
plot_acf(data_ts, lags=50)
plt.title('Autocorrelation Function (ACF) - Google Close Price')
plt.show()
plt.figure(figsize=(12, 5))
plot_pacf(data_ts, lags=50)
plt.title('Partial Autocorrelation Function (PACF) - Google Close Price')
plt.show()
```
Split data:
```
train_size = int(len(data_ts) * 0.8)
train, test = data_ts[:train_size], data_ts[train_size:]
```
Fit SARIMA model:
```
print("\n--- Fitting SARIMA(1, 1, 1)x(1, 1, 1, 5) model ---")
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5), enforce_stationarity=False, enforce_invertibility=False)
sarima_result = sarima_model.fit(disp=False)
```
Make predictions& Evaluate RMSE:
```
predictions = sarima_result.predict(start=len(train), end=len(data_ts) - 1) 

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)
```
Plot Predictions:
```
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.title(f'SARIMA Model Predictions (RMSE: {rmse:.3f})')
plt.legend()
plt.grid(True)
plt.show()
```

### OUTPUT:
Original Data:

<img width="887" height="476" alt="image" src="https://github.com/user-attachments/assets/4d640efe-27d3-4e94-bf8b-3b13b059d0d5" />

Autocorrelation:

<img width="742" height="541" alt="image" src="https://github.com/user-attachments/assets/847aae30-ad91-43a3-936b-6e2cd94ca740" />

Partial Autocorrelation:

<img width="719" height="540" alt="image" src="https://github.com/user-attachments/assets/61a3d188-42a3-4933-829e-a441c475552a" />

SARIMA Model:

<img width="892" height="478" alt="image" src="https://github.com/user-attachments/assets/3f4b987f-3388-45ed-a1b6-3c226cb18285" />

RMSE Value:

<img width="236" height="32" alt="image" src="https://github.com/user-attachments/assets/a64d71e3-10f3-4ec8-9d41-2f705ee8d536" />

### RESULT:
Thus the program run successfully based on the SARIMA model.
