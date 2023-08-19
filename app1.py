import numpy as np
from flask import Flask, request, render_template
import pickle



app = Flask(__name__,template_folder='templates')

model = pickle.load(open('model_1.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index1.html')



@app.route('/predict',methods=['post'])
def predict():
   
    int_features = [float(x) for x in request.form.values()] #Convert string inputs to float.
    features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
    prediction = model.predict(features)  # features Must be in the form [[a, b]]

    output = round(prediction[0], 2)
    return render_template('index1.html', prediction_text='Score  is {}'.format(output))


if __name__ == '__main__' :
    app.run()