import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# load data
df = pd.read_csv("kiva_loans.csv")

# create dataframe
df = df[['funded_amount', 'loan_amount', 'activity', 'sector',  'country_code', 'country',
         'region', 'term_in_months', 'lender_count', 'borrower_genders',
         'repayment_interval']]

# drop missing vals
df = df.dropna()

# gender -> categorical value
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
X = df.drop(['loan_amount', 'funded_amount','country_code'], axis=1)
y = df['loan_amount']

# normalize
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# network archeticture
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# compile
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

# fit
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# predict
y_pred = model.predict(X_test).flatten()

# evaluate
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error:", mse)
r2 = r2_score(y_test, y_pred)
print("R-squared value:", r2)

# plot predictions vs actual
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot(y_test, y_test, color='red', label='Ideal Predictions')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Loan Amount")
plt.legend()
plt.xlim(0, 10000)
plt.ylim(0, 10000)
plt.show()

# calculate permutation importances
result = permutation_importance(model, X_test, y_test, scoring='neg_mean_squared_error', n_repeats=10, random_state=42)
perm_importances = result.importances_mean

# sort permutationed importances
sorted_index = np.argsort(perm_importances)[::-1]
sorted_importances = perm_importances[sorted_index]
sorted_features = X.columns[sorted_index]

# plot feature importances
plt.figure(figsize=(14, 8))
plt.bar(range(len(sorted_features)), sorted_importances, tick_label=sorted_features)
plt.title("Feature Importances (Permutation Importance)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

