import pandas as pd
import pickle


# load the model from disk
model_filename = 'Files/finalized_model.pkl'
final = pickle.load(open(model_filename, 'rb'))
label_filename = 'Files/label_encoder.pkl'
le = pickle.load(open(label_filename, 'rb'))
scaling_filename = 'Files/scaling.pkl'
scaler = pickle.load(open(scaling_filename, 'rb'))

def preprocess_test_data(test):
    test = test.drop(['Survived','PassengerId','Name','Ticket','Cabin','Embarked'], axis=1)
    test['Sex']= le.transform(test['Sex'])
    test=scaler.transform(test)
    test.to_csv('Files/test.csv')

def inference_prediction(test):
    test['Sex']= le.transform(test['Sex'])
    test=scaler.transform(test)
    y=final.predict(test)
    return y

