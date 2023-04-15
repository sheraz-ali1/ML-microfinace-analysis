from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# define 3 models
model1 = RandomForestRegressor(n_estimators=100, random_state=42)
model2 = RandomForestRegressor(n_estimators=100, random_state=43)
model3 = RandomForestRegressor(n_estimators=100, random_state=44)
# load data
df = pd.read_csv("kiva_loans.csv")

# create dataframe
df = df[['funded_amount', 'loan_amount', 'activity', 'sector',  'country_code', 'country',
         'region', 'term_in_months', 'lender_count', 'borrower_genders',
         'repayment_interval']]

# gender -> numerical val
df['borrower_genders'] = df['borrower_genders'].astype(str)
df['borrower_genders'] = df['borrower_genders'].apply(lambda x: 1 if 'female' in x else 0)


# categorical -> numerical
le = LabelEncoder()
df['activity'] = le.fit_transform(df['activity'])
df['sector'] = le.fit_transform(df['sector'])
df['country_code'] = le.fit_transform(df['country_code'])
df['country'] = le.fit_transform(df['country'])
df['region'] = le.fit_transform(df['region'])
df['repayment_interval'] = le.fit_transform(df['repayment_interval'])

# drop missing vals
df = df.dropna()

# features and targets
X = df.drop(['loan_amount', 'funded_amount','country_code'], axis=1)
y = df['loan_amount']


# normalize
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)


# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train all 3 models
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

# predict
p1 = model1.predict(X_test)
p2 = model2.predict(X_test)
p3 = model3.predict(X_test)

# intialize
w1 = 1/3
w2 = 1/3
w3 = 1/3

arr = []
# loop
for i in range(len(X_test)):
    
    # combine
    prediction = w1*p1[i] + w2*p2[i] + w3*p3[i]
    
    # error
    error1 = abs(p1[i] - y_test.iloc[i])
    error2 = abs(p2[i] - y_test.iloc[i])
    error3 = abs(p3[i] - y_test.iloc[i])
    
    # normalize errors
    error_sum = error1 + error2 + error3
    error1 = error1 / error_sum
    error2 = error2 / error_sum
    error3 = error3 / error_sum
    
    # update
    w1 = w1 / 2 ** error1
    w2 = w2 / 2 ** error2
    w3 = w3 / 2 ** error3
    
    # sum
    w_sum = w1 + w2 + w3
    w1 = w1 / w_sum
    w2 = w2 / w_sum
    w3 = w3 / w_sum

    arr.append(prediction)

# predict
final_predictions = w1 * model1.predict(X_test) + w2 * model2.predict(X_test) + w3 * model3.predict(X_test)

# evaluate
mse = mean_squared_error(y_test, final_predictions)
r2 = r2_score(y_test, final_predictions)


print(final_predictions)
print("Mean squared error: {:.2f}".format(mse))
print("R2 score: {:.2f}".format(r2))
