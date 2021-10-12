#Dependencies and Setup
import numpy as np
from flask import Flask, request, render_template
import pickle


# TO LOAD THE MODEL AND THE SCALER!
app = Flask(__name__)
loan_model = pickle.load(open('Resources/RFCmodel.pkl', 'rb'))
# loan_scaler = pickle.load(open('Resources/scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    #For rendering results on HTML 
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print (final_features) 
    # scaled_features = loan_scaler.transform(final_features)
    prediction = loan_model.predict(final_features)
    print(prediction)
    classes = np.array(["loan can be fully paid", "you may not be able to pay off the loan"])

    output = classes[prediction][0]

    return render_template('index.html', prediction_text='The predicted result is: {}'.format(output))
    

if __name__ == "__main__":
    app.run(debug=True)