import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)
   
    
# POSTMAN
# {
#     "data":{
#     "CRIM": 0.00632,
#     "ZN": 18.0,
#     "INDUS": 2.31,
#     "CHAS": 0.0,
#     "NOX": 0.538,
#     "RM": 6.575,
#     "Age": 65.2,
#     "DIS": 4.0900,
#     "RAD": 1.0,
#     "TAX": 296,
#     "PTRATIO": 15.3,
#     "B": 396.90,
#     "LSTAT": 4.98
# }
# }

