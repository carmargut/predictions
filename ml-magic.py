#Jesus Copado y Carlos Martinez

# LIBRARIES

import pandas
import numpy as np
# Import the linear regression class
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from sklearn.externals import joblib


# FUNCTIONS

def map_operator(op):
	if op == '+':
		return 0
	if op == '-':
		return 1
	if op == '*':
		return 2
	return 3 #/

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
	if x < 10:
		return 1
	if x % 10 == 0:
		return 2
	return 3

'''
def get_operand_group(x): 
	if x == 0:
		return 0
	if x == 1:
		return 1
	if x < 10:
		return 2
	if x % 10 == 0:
		return 3
	if x == 11:
		return 4
	return 5
    
def get_operand_group(x):
	if x == 1:
		return 0
	if x < 10:
		return 1
	if x == 10:
		return 2
	if x == 11:
		return 3
	if x % 10 == 0:
		return 4
	if x % 5 == 0:
		return 5
	if x < 20:
		return 6
	if x < 50:
		return 7
	return 8
'''

def is_first_smaller(row):
	if row.op1 < row.op2:
		return 1
	return 0

def get_dec(x):
	return x // 10

def get_uni(x):
	return x % 10


#LECTURA DEL ARCHIVO

operations = pandas.read_csv("allData.csv", names=['name', 'op1', 'op', 'op2', 'time'])


#FEATURE TRANSFORMATION

#operations['is_easy'] = operations.apply(is_easy, axis=1)
#operations['is_difficult'] = operations.apply(is_difficult, axis=1)
#operations['difficulty'] = operations.apply(get_difficulty, axis=1)

operations['is_first_smaller'] = operations.apply(is_first_smaller, axis=1)

operations.op = operations.op.apply(map_operator)

operations['op1_group'] = operations.op1.apply(get_operand_group)
operations['op2_group'] = operations.op2.apply(get_operand_group)

operations['dec1'] = operations.op1.apply(get_dec)
operations['uni1'] = operations.op1.apply(get_uni)
operations['dec2'] = operations.op2.apply(get_dec)
operations['uni2'] = operations.op2.apply(get_uni)

#LIMPIEZA DE DATOS MALIGNOS

operations = operations[operations.time > 0] # por si hay valores negativos
operations = operations[operations.time < 120] # despreciar tiempos demasiado altos
operations = operations[(operations.op1 != 0) & (operations.op2 != 0)] # no debería haber ceros
operations = operations[(operations.time > 10) | (operations.op1_group != 3) | (operations.op2_group != 3)
                        | (operations.op != 2)] # multiplicaciones complicadas con un tiempo sospechasemente bajo


#MACHINE LEARNING MAGIC

lm = RandomForestRegressor(n_estimators=80, max_depth=4.5)
lm2 = DecisionTreeRegressor(max_depth=4.5)
lm3 = linear_model.Ridge(alpha = .5)
lm4 = linear_model.LinearRegression()

# The columns we'll use to predict the target
predictors = ['op1', 'op2', 'is_first_smaller', 'op']

X = operations[predictors]
y = operations.time

lm.fit(X, y)
print (np.sqrt(-cross_val_score(lm, X, y, cv=10, scoring='mean_absolute_error')).mean())
#2.362


#EXPORT MODEL INTO A FILE
joblib.dump(lm, 'Regressor.pkl')