#Jesus Copado y Carlos Martinez

'''
FUNCIONES INTERESANTES:

PARA HACER UN MAPPING DE CADA VALOR A UN ENTERO VALE CON ESTA FUNCION:
le = preprocessing.LabelEncoder()
operations.sex = le.fit_transform(operations.sex)

PARA SABER EL NUMERO DE VECES QUE APARECE UN FEATURE:
print(pandas.value_counts(operations["marital-status"]))

print(titanic["Sex"].unique())

loc para buscar
print(operations.loc[operations['time'] < 0])

para predecir a pelo
lm.predict([[98, 83, 2]])
'''

import re
import operator
import pandas
import numpy as np
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest, f_classif
# Import the linear regression class
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing
import matplotlib.pyplot as plt

from flask import Flask
from sklearn.externals import joblib


def map_operator(op):
	if op == '+':
		return 0
	if op == '-':
		return 1
	if op == '*':
		return 2
	return 3 #/

def map_operand(x):
	if x < 10:
		return 0
	return 1

def map_time(t):
	if t < 1:
		return 0
	return 1

def is_easy(row): #ampliar
	if row.op == '/' and row.op1 < row.op2:
		return 1
	if row.op == '*' and (row.op1 == 1 or row.op2 == 1):
		return 1
	if row.op == '*' and (row.op1 == 0 or row.op2 == 0):
		return 1
	return 0
	
def is_difficult(row): #ampliar
	if row.op == '-' and row.op1 > row.op2:
		return 1
	if row.op == '*' and (row.op1 > 10 or row.op2 > 10) and (row.op1 % 10 != 0 and row.op2 % 10 != 0):
		return 1
	return 0

#0 muy facil - 4 muy dificil
def get_difficulty(row): 
	if row.op == '*':
		if row.op1 == 1 or row.op2 == 1:
			return 0
	if row.op == '/':
		if row.op1 == 1 or row.op2 == 1:
			return 0
	if row.op == '+':
		if row.op1 == 1 or row.op2 == 1:
			return 0
	if row.op == '-':
		if row.op1 < row.op2:
			return 3
	return 2


def get_operand_group(x): 
	if x == 0:
		return 0
	if x == 1:
		return 1
	if x < 10:
		return 2
	if x % 10 == 0:
		return 3
	return 4


def is_first_smaller(row):
	if row.op1 < row.op2:
		return 1
	return 0

operations = pandas.read_csv("allData.csv", names=['name', 'op1', 'op', 'op2', 'time'])
operations = operations[operations.time > 0] # por si hay valores negativos
#operations = operations[operations.time < 300] # despreciar tiempos demasiado altos

operations['difficulty'] = operations.apply(get_difficulty, axis=1)
operations['is_easy'] = operations.apply(is_easy, axis=1)
operations['is_difficult'] = operations.apply(is_difficult, axis=1)
operations['is_first_smaller'] = operations.apply(is_first_smaller, axis=1)

operations.op = operations.op.apply(map_operator)
operations['op1_group'] = operations.op1.apply(get_operand_group)
operations['op2_group'] = operations.op2.apply(get_operand_group)

lm = DecisionTreeRegressor(max_depth=5)
lm2 = linear_model.Ridge(alpha = .5)
lm3 = linear_model.LinearRegression()

'''
feature_cols = ['op1', 'op2', 'op1_group', 'op2_group', 'op']
X = operations[feature_cols]
y = operations.time

lm.fit(X, y)
lm.predict([[94, 92, 4, 4, 2]])
'''

# The columns we'll use to predict the target
predictors = ['op1_group', 'op2_group', 'op']

# Initialize the cross validation folds
kf = KFold(operations.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
	train_target = operations.time.iloc[train]
	# Fit the algorithm on the training data.
	lm.fit(operations[predictors].iloc[train,:], train_target)
	# Select and predict on the test fold.
	# The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
	test_predictions = lm.predict(operations[predictors].iloc[test,:])
	predictions.append(test_predictions)

# Put all the predictions together into one array.
predictions = np.concatenate(predictions, axis=0)

error = abs(operations.time - predictions)
print('Porcentaje de aciertos (error 5 segundos): ', sum(error < 5) / len(predictions))
print('Media del error: ', error.mean())

X = operations[predictors]
y = operations.time
scores = cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')
print(scores.mean())

app = Flask(__name__)

# Cargo el modelo que he entrenado previamente.
#regr = joblib.load('filename.pkl') 


@app.route("/")
def hello():
    return "Hello World!"

@app.route("/predict/<int:op1>/<int:op>/<int:op2>/")
def predict(op1, op, op2):
	n1 = get_operand_group(op1)
	n2 = get_operand_group(op2)
	return str(lm.predict([[n1, n2, op]])[0])

@app.route("/pred/<int:op1>/")
def pred(op1):
	n1 = get_operand_group(op1)
	return str(n1)

if __name__ == "__main__":
	app.run()