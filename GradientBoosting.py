import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

# load data
df = pd.read_csv("kiva_loans.csv")

# create dataframe
df = df[['funded_amount', 'loan_amount', 'activity', 'sector', 'country_code', 'country',
         'region', 'term_in_months', 'lender_count', 'borrower_genders',
         'repayment_interval']]

# drop missing vals
df = df.dropna()

# gender -> numerical value
df['borrower_genders'] = df['borrower_genders'].apply(lambda x: 1 if 'female' in x else 0)

# categorical values -> numerical values
le = LabelEncoder()
df['activity'] = le.fit_transform(df['activity'])
df['sector'] = le.fit_transform(df['sector'])
df['country_code'] = le.fit_transform(df['country_code'])
df['country'] = le.fit_transform(df['country'])
df['region'] = le.fit_transform(df['region'])
df['repayment_interval'] = le.fit_transform(df['repayment_interval'])

# features and targets
X = df.drop(['loan_amount', 'funded_amount', 'country_code'], axis=1)
y = df['loan_amount']

# normalize features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# train the gradient boosting regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# predict
y_pred = gb_model.predict(X_test)

# evaluate
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error:", mse)
r2 = r2_score(y_test, y_pred)
print("R-squared value:", r2)

# Get the feature importances from the Gradient Boosting Regressor model
importances = gb_model.feature_importances_

# sort by feature importance
sorted_index = np.argsort(importances)[::-1]
sorted_importances = importances[sorted_index]
sorted_features = X.columns[sorted_index]

# plot bar chart
plt.figure(figsize=(14, 8))
plt.bar(range(len(sorted_features)), sorted_importances, tick_label=sorted_features, color='black', alpha=0.8)
plt.title("Feature Importances - Gradient Boosting Regressor", fontsize=14, fontweight='bold')
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()
