import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# load data
df = pd.read_csv("kiva_loans.csv")

# create data frame
df = df[['funded_amount', 'loan_amount', 'activity', 'sector',  'country_code', 'country',
         'region', 'term_in_months', 'lender_count', 'borrower_genders',
         'repayment_interval']]

# drop missing values
df = df.dropna()

# gender -> binary value
df['borrower_genders'] = df['borrower_genders'].apply(lambda x: 1 if 'female' in x else 0)

# label encoder to convert categorical values to numerical
le = LabelEncoder()
df['activity'] = le.fit_transform(df['activity'])
df['sector'] = le.fit_transform(df['sector'])
df['country_code'] = le.fit_transform(df['country_code'])
df['country'] = le.fit_transform(df['country'])
df['region'] = le.fit_transform(df['region'])
df['repayment_interval'] = le.fit_transform(df['repayment_interval'])

# Split the data into X and y
X = df.drop(['loan_amount', 'funded_amount','country_code'], axis=1)
y = df['loan_amount']

# normalize features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# train model
lr = LinearRegression()
lr.fit(X_train, y_train)

# predict
y_pred = lr.predict(X_test)

# evaluate
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error:", mse)
r2 = r2_score(y_test, y_pred)
print("R-squared value:", r2)
print("Coefficients: \n", lr.coef_)

import matplotlib.pyplot as plt

# plot the regression predictions
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot(y_test, y_test, color='red', label='Ideal Predictions')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Loan Amount",fontsize=12, fontweight='bold')
plt.legend()
plt.xlim(0, 10000)
plt.ylim(0, 10000)
plt.show()

# derive coeffeciants
coef = lr.coef_
features = X.columns

# abs value coeffeciants
abs_coef = np.abs(coef)

# normalize coeffeciants
norm_coef = abs_coef / np.sum(abs_coef)

# sort coeffeciants
sorted_index = np.argsort(norm_coef)[::-1]
sorted_coef = norm_coef[sorted_index]
sorted_features = features[sorted_index]

# bar chart
plt.figure(figsize=(14, 8))
plt.bar(range(len(sorted_features)), sorted_coef, tick_label=sorted_features, color = 'black', alpha = 0.8)
plt.title("Feature Importances - Linear Regression",fontsize=14, fontweight='bold')
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()
