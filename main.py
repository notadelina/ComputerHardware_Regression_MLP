from math import sqrt
import numpy as np
import pandas as pd
from sklearn import neural_network
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# INCARCARE DATE
data = pd.read_csv('machinee.csv', na_values=[''], na_filter=False)
data = data.dropna()
le = LabelEncoder()

# Fit the encoder to all the columns that contain categorical data
for col in data.columns:
    if data[col].dtype == 'object':
        le.fit(data[col])
        data[col] = le.transform(data[col])

data['PRP'] = data['PRP'].astype('category')
categ = np.array(data['PRP'].cat.codes.values).reshape(-1, 1)
conts = np.array(data.drop(['PRP'], axis=1))

y = np.array(data['PRP'].values)

size = len(data['PRP'])
train_size = int(0.75 * size)
test_size = int(0.25 * size)

# IMPARTIRE IN TRAIN SI TEST
categ_train = categ[:train_size]
categ_test = categ[train_size:]

cont_train = conts[:train_size]
cont_test = conts[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]


# print('Date de antrenare:\n', np.append(categ_train,cont_train,axis=1))
# print('Etichete de antrenare:\n', y_train)
# print('\nDate de testare:\n', np.append(categ_test,cont_test,axis=1))
# print('Etichete de testare:\n', y_test)

# CREARE SI ANTRENARE MLP



# 1 strat si learning rate = 0.1

regr = neural_network.MLPRegressor(hidden_layer_sizes=(15,7),max_iter=2000,learning_rate_init=0.1)
regr.fit(np.append(categ_train,cont_train,axis=1),y_train)
predictii = regr.predict(np.append(categ_test,cont_test,axis=1))

# EROARE
MSE = mean_squared_error(predictii,y_test)
print('\nRMSE: ', str(sqrt(MSE)))


# 2 straturi si learning rate = 0.1

regr = neural_network.MLPRegressor(hidden_layer_sizes=(10,10),max_iter=2000,learning_rate_init=0.1)
regr.fit(np.append(categ_train,cont_train,axis=1),y_train)
predictii = regr.predict(np.append(categ_test,cont_test,axis=1))

# EROARE
MSE = mean_squared_error(predictii,y_test)
print('\nRMSE: ', str(sqrt(MSE)))

# 2 straturi si learning rate = 0.1

regr = neural_network.MLPRegressor(hidden_layer_sizes=(10,5),max_iter=2000,learning_rate_init=0.1)
regr.fit(np.append(categ_train,cont_train,axis=1),y_train)
predictii = regr.predict(np.append(categ_test,cont_test,axis=1))

# EROARE
MSE = mean_squared_error(predictii,y_test)
print('\nRMSE: ', str(sqrt(MSE)))



# 1 strat si learning rate = 0.01

regr = neural_network.MLPRegressor(hidden_layer_sizes=10,max_iter=2000,learning_rate_init=0.01)
regr.fit(np.append(categ_train,cont_train,axis=1),y_train)
predictii = regr.predict(np.append(categ_test,cont_test,axis=1))

# EROARE
MSE = mean_squared_error(predictii,y_test)
print('\nRMSE: ', str(sqrt(MSE)))


# 2 straturi si learning rate = 0.01

regr = neural_network.MLPRegressor(hidden_layer_sizes=(10,10),max_iter=2000,learning_rate_init=0.01)
regr.fit(np.append(categ_train,cont_train,axis=1),y_train)
predictii = regr.predict(np.append(categ_test,cont_test,axis=1))

# EROARE
MSE = mean_squared_error(predictii,y_test)
print('\nRMSE: ', str(sqrt(MSE)))

# 2 straturi si learning rate = 0.01
regr = neural_network.MLPRegressor(hidden_layer_sizes=(10,5),max_iter=2000,learning_rate_init=0.01)
regr.fit(np.append(categ_train,cont_train,axis=1),y_train)

# TESTARE MLP
predictii = regr.predict(np.append(categ_test,cont_test,axis=1))


# for i in range(test_size):
#     print(f"Valoarea prezisa: {predictii[i]}  Valoarea reala: {y_test[i]}")


# EROARE
MSE = mean_squared_error(predictii,y_test)
print('\nRMSE: ', str(sqrt(MSE)))