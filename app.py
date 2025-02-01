import pickle

from flask import Flask,request,render_template
import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        # Extract data from the form
        data = CustomData(
            UDI=int(request.form['UDI']),
            Type=str(request.form['Type']),
            air_temperature=float(request.form['air_temperature']),
            process_temperature=float(request.form['process_temperature']),
            rotational_speed=float(request.form['rotational_speed']),  # Correct field name
            torque=float(request.form['torque']),
            tool_wear=int(request.form['tool_wear']),
            target=int(request.form['target'])
        )

        # Convert data to DataFrame
        pred_df = data.get_data_as_data_frame()

        # Make predictions
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        mapping={
            0:'No Failure', 
            1:'Power Failure', 
            2:'Tool Wear Failure',
            3:'Overstrain Failure',
            4:'Random Failures',
            5:'Heat Dissipation Failure'
        }
        result=mapping.get(results[0])
        # Render the result in the HTML template
        return render_template('index.html', results=result)
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)