# Importing Necessary Modules
from flask import *
import inference_pipeline
import json
import pandas as pd

app = Flask(__name__)
 
# Create a Main route here
@app.route('/home')
def input():
     return('Hello World')

    
@app.route('/power', methods=['POST'])
def post_request():
    number = request.form.get('number1')
    return jsonify({'result':int(number)**2})

@app.route('/survival_prediction', methods=['POST'])
def post_request_titanic():
    Pclass = request.form.get('Pclass')
    Sex = request.form.get('Sex')
    Age = request.form.get('Age')
    SibSp = request.form.get('SibSp')
    Parch = request.form.get('Parch')
    Fare = request.form.get('Fare')

    dict1= {'Pclass':int(Pclass),'Sex':Sex,'Age':float(Age),'SibSp':int(SibSp),'Parch':int(Parch),'Fare':float(Fare)} 
    test_sample_df = pd.DataFrame(dict1,index = [0])
    

    result = inference_pipeline.inference_prediction(test_sample_df)
    if result[0]==1:
        return json.dumps({'result':'Survived'})
    else:
        return json.dumps({'result':'Not Survived'})

   



# main route to start with
if __name__ == '__main__':
    app.run(debug=True)