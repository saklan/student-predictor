from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return "🎓 Student Predictor is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    hours = data['hours']
    pred = model.predict([[hours]])
    return jsonify({'prediction': round(pred[0], 2)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
