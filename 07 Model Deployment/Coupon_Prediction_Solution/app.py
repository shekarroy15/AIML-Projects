import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('rf_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print(request.form)
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    # output = round(prediction[0], 2)
    #print(prediction)
    if prediction == 1:
        return render_template('index.html', _anchor="predform", prediction_text='Hurrah..You have got a discount of 25%' )
    else:
        return render_template('index.html', _anchor="predform", prediction_text='Sorry...You will not get any discount')

if __name__ == "__main__":
    app.run(debug=True)

## Source for index.html: https://www.free-css.com/