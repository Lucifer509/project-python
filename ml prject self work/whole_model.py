import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from joblib import load, dump


housing = pd.read_csv('data.csv')
# print(housing)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):         # we made ratio in according to 'CHAS'
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing_analise = strat_test_set
housing_train = strat_train_set.drop('MEDV', axis=1)
housing_train_labels = strat_train_set['MEDV']


# main model start from here ...
my_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scalar',StandardScaler())
])

housing_prossed = my_pipeline.fit_transform(housing_train)

model = RandomForestRegressor()

some_feature = housing_train.iloc[:5]
some_labels = housing_train_labels.iloc[:5]

some_prossed_data = my_pipeline.fit_transform(some_feature)

model.fit(some_prossed_data, some_labels)
prediction = model.predict(some_prossed_data)
# print(prediction)
# print(some_labels)

model.fit(housing_prossed, housing_train_labels)

housing_predict = model.predict(housing_prossed)
mse = mean_squared_error(housing_train_labels, housing_predict)
rmse = np.sqrt(mse)

print("Take a small part of large data to see model working", rmse)

score = cross_val_score(model, housing_prossed, housing_train_labels, scoring='neg_mean_squared_error', cv=10)
rmse_score = np.sqrt(-score)

print("rmse after training model through cross_val_score method:", rmse_score)

# save the model
dump(model, 'final_model.joblib')

# test model
X_test_feature = strat_test_set.drop("MEDV", axis=1)
X_test_labels = strat_test_set["MEDV"]

X_prepare = my_pipeline.fit_transform(X_test_feature)

final_predict = model.predict(X_prepare)

final_mse = mean_squared_error(X_test_labels, final_predict)
final_rmse = np.sqrt(final_mse)

print("final rmse at testing:", final_rmse)


# plot some in site data ....
housing_analise.hist(bins=50, figsize=(20,15))

attributes = ['MEDV', "RM", 'LSTAT', 'B']
scatter_matrix(housing_analise[attributes], figsize=(12, 10))
plt.show()




