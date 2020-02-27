#Jesus Copado y Carlos Martinez

from flask import Flask
from sklearn.externals import joblib

def is_first_smaller(op1, op2):
	if op1 < op2:
		return 1
	return 0
	
app = Flask(__name__)

regr = joblib.load('Regressor.pkl')

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/predict/<int:op1>/<int:op>/<int:op2>/")
def predict(op1, op, op2):
	s = is_first_smaller(op1, op2)
	return str(regr.predict([[op1, op2, s, op]])[0])

if __name__ == "__main__":
	app.run()