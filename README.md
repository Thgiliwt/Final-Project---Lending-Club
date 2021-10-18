# Final Project - Loan Prediction ML

PPT Slices: https://docs.google.com/presentation/d/17eyFWEB0vS2pABSSRgFKjQkA8fTW2gmEoVFMNe3UlW0/edit?usp=sharing

Website: https://lendingclub-machinelearning.herokuapp.com/


# Data Sources

LENDING CLUB: https://en.wikipedia.org/wiki/LendingClu
Kaggle: https://www.kaggle.com/wendykan/lending-club-loan-data/download%E2%80%9D

# Coding Appraoch

Python -to clean  and preprocess the dataset, to test machine learning models and save the best ml model for future predictions.
Flask - to load the machine learning model saved, and apply the model to the data that input from the website, predict the result and render onto the webpage
HTML - to setup the structure of the webpage and display the loan calculator
CSS- To set the style of the webpage
Heroku- to save and showcase the webpage

# Models and Results

Highest F1 scores
Logistic Regression: 0.9167

Other model tested:
KNN Classifier
Random Forest Classifier
Decision Tree Classifier

# Challenges

Extremely imbalance label, around 6:1

Have not applied every numeric feature to predict (originally around 150 features)

LR is performing better than the others in our case, but may not be the best practice, have not tried neural networks in keras, Naive Bayes, XGBoost

How to encoding, scaling and labeling variables based on the data nature (loan status have 7, but we only use 2)

Data completeness and accuracy is compromised through data cleaning /pre-processing part
