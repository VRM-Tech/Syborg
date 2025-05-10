from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Simulate the same dataset generation and model training as your code
n_samples = 5000
np.random.seed(42)
labels = np.random.choice([0, 1], size=n_samples, p=[0.80, 0.20])

def generate_features(label):
    if label == 0:
        return [np.random.normal(60, 20), np.random.normal(60, 20), np.random.poisson(40),
                np.random.poisson(35), np.random.poisson(25), np.random.poisson(20),
                np.random.normal(10000, 3000)]
    else:
        return [np.random.normal(5, 2), np.random.normal(5, 2), np.random.poisson(100),
                np.random.poisson(90), np.random.poisson(70), np.random.poisson(65),
                np.random.normal(100, 50)]

data = [generate_features(label) for label in labels]
columns = [
    'Avg min between sent tnx',
    'Avg min between received tnx',
    'Sent tnx',
    'Received Tnx',
    'Unique Sent To Addresses',
    'Unique Received From Addresses',
    'Time Diff between first and last (Mins)'
]

df = pd.DataFrame(data, columns=columns)
df['Label'] = labels

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
iso_forest = IsolationForest(n_estimators=100)
iso_forest.fit(X_scaled)

@app.route('/')
def serve_html():
    return send_from_directory('.', 'user_input_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array(data['input']).reshape(1, -1)

    input_scaled = scaler.transform(input_data)
    rf_prediction = rf.predict(input_data)[0]
    iso_prediction = iso_forest.predict(input_scaled)[0]

    if iso_prediction == -1:
    if rf_prediction == 1:
        result = "Sybil"
    else:
        result = "Not Sybil, but Anamoly"
    else:
        result = "Normal"

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)