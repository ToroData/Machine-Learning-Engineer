from flask import Flask, session, jsonify, request
import json
import os
import pandas as pd
import diagnosis
from scoring import score_model



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    if request.method == 'POST':
        file = request.files['filename']
        dataset = pd.read_csv(file)
        return model_predictions(dataset)
    if request.method == 'GET':
        file = request.args.get('filename')
        dataset = pd.read_csv(file)
        return {'predictions': str(model_predictions(dataset))}

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats():        
    #check the score of the deployed model
    return {'F1 score': score_model()}

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    summary_data = diagnostics.dataframe_summary()
    return summary_data

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def stats():        
    #check timing and percent NA values
    missing_data_percentages = diagnostics.missing_data()
    timing = diagnostics.execution_time()
    return [timing, missing_data_percentages]

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
