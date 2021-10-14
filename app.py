#Dependencies and Setup
import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  precision_recall_curve, roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score,auc, roc_curve, plot_confusion_matrix




# TO LOAD THE MODEL AND THE SCALER!
app = Flask(__name__)
loan_model = pickle.load(open('lrmodel.pkl', 'rb'))
# loan_scaler = pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    #For rendering results on HTML 
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    # print (final_features) 
    # scaled_features = loan_scaler.transform(final_features)
    prediction = loan_model.predict(final_features)
    # print(prediction)
    classes = np.array(["you may not be able to pay off the loan","loan can be fully paid"])

    output = classes[prediction][0]

    return render_template('index.html', prediction_text='The predicted result is: {}'.format(output))
    

if __name__ == "__main__":
    app.run(debug=True)