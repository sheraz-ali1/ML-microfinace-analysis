import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

# load data
kiva_loans = pd.read_csv("kiva_loans.csv")

# convert to datetime format
kiva_loans['posted_time'] = pd.to_datetime(kiva_loans['posted_time'])
kiva_loans.set_index('posted_time', inplace=True)

# aggregate loan $ by month
monthly_loan_amount = kiva_loans['loan_amount'].resample('M').sum()

# decompose the time series visulize patterns in seasonality
decomposition = seasonal_decompose(monthly_loan_amount)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
decomposition.trend.plot(ax=ax1, title='Trend')
decomposition.seasonal.plot(ax=ax2, title='Seasonality')
decomposition.resid.plot(ax=ax3, title='Residuals')
plt.tight_layout()
plt.show()

# fit the SARIMAX model
sarimax_model = SARIMAX(monthly_loan_amount, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarimax_results = sarimax_model.fit()

# forecast
forecast_steps = 12
forecast = sarimax_results.get_forecast(steps=forecast_steps)
forecast_ci = forecast.conf_int()

# plot forecasts
plt.figure(figsize=(12, 6))
plt.plot(monthly_loan_amount, label='Observed')
plt.plot(forecast.predicted_mean, label='Forecast', color='r')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink')
plt.xlabel('Date')
plt.ylabel('Loan Amount')
plt.legend()
plt.show()
