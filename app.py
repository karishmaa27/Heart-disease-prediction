from heart_disease_prediction import SVM_Classifier
import pickle
from flask import Flask, render_template, request
import numpy as np

model = pickle.load(open('heart_svm.pkl', 'rb'))


app = Flask(__name__, static_url_path='/static')

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert form inputs to float
        d1 = float(request.form['a'])
        d2 = float(request.form['b'])
        d3 = float(request.form['c'])
        d4 = float(request.form['d'])
        d5 = float(request.form['e'])
        d6 = float(request.form['f'])
        d7 = float(request.form['g'])
        d8 = float(request.form['h'])
        d9 = float(request.form['i'])
        d10 = float(request.form['j'])
        d11 = float(request.form['k'])
        d12 = float(request.form['l'])
        d13 = float(request.form['m'])

        arr = np.array([[d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13]])
        pred = model.predict(arr)
        return render_template('after.html', data=pred)
    except ValueError as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
