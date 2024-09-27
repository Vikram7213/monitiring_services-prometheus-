from flask import Flask, request, jsonify
import joblib
import pandas as pd
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)

# Initialize Prometheus metrics
metrics = PrometheusMetrics(app)

# Load the pre-trained model
model = joblib.load('iris_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json
    df = pd.DataFrame(data)
    
    # Make a prediction
    predictions = model.predict(df)
    
    # Return the prediction as a JSON response
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
