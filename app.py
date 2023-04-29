import numpy as np
import pandas as pd
import xgboost
import pickle
from flask import Flask, render_template, request
import sklearn

app = Flask(__name__)

model= pickle.load(open('model.pkl', 'rb'))
preprocessor=pickle.load(open('preprocessor.pkl', 'rb'))

@app.route('/')
def Home():
    return render_template(('index.html'))

@app.route('/', methods=['POST'])
def predict():
    # gathering inputs
    account_length= int(request.form.get('account_length'))
    international_plan= request.form.get('international_plan')
    vmail_message=int(request.form.get('vmail_message'))
    day_calls=int(request.form.get('day_calls'))
    day_charge=float(request.form.get('day_charge'))
    eve_charge=float(request.form.get('eve_charge'))
    night_charge=float(request.form.get('night_charge'))
    international_calls=int(request.form.get('international_calls'))
    international_charge=float(request.form.get('international_charge'))
    custServ_calls=int(request.form.get('custServ_calls'))

    inputs= np.array([account_length, international_plan, vmail_message, day_calls,
                      day_charge, eve_charge, night_charge, international_calls,
                      international_charge, custServ_calls]).reshape(1,-1)

    input_processed= preprocessor.transform(inputs)

    prediction= model.predict(input_processed)

    # Generate churn risk scores
    churn_risk_scores = model.predict_proba(input_processed)[:, 1]*100

    #churn flag
    if prediction==1:
        prediction='YES'

    else:
        prediction='NO'

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
